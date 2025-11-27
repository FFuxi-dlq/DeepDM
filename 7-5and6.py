from jarvis.core.specie import Specie,get_node_attributes
from ase.atoms import Atom
from e3nn.nn import Gate
from e3nn.util.jit import compile_mode
import math
from ase.data import atomic_numbers
from torch_geometric.loader import DataLoader as PyGDataLoader
import gc
import warnings
warnings.filterwarnings("ignore")
import torch
from e3nn import o3, nn
from torch_geometric.data import Data
import warnings
import torch.nn.functional as F
from e3nn.o3 import Irreps
from typing import Dict, Union,Tuple
from e3nn.math import soft_one_hot_linspace, soft_unit_step
import time
import numpy as np
import torch
import e3nn
from torch import nn as torch_nn
from e3nn import nn as e3nn_nn

from ase import Atoms
from torch_geometric.data import Data

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import degree
import ase.neighborlist

from pyscf.pbc import gto
import numpy as np

from itertools import product
from typing import List
import random
from tqdm import tqdm


seed = 42 # 可以是任何整数
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 为了完全的确定性，有时还需要以下设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
basis = 'gth-szv'
#basis= 'sto_3g'
pseudo = 'gth-pade'
#pseudo=None
xc = 'lda,vwn'
batch_size=1
# --- Parameters ---
lmax_sh = 4
heads = 1
radius=10
r_cut =10
layers =2

B2A = 0.529177
A2B = 1.889726

class trans():
    def __init__(self, cell: gto.Cell, k_mesh: List, dimension:int,type=torch.complex128):
        self.cell=cell
        self.kpts=torch.from_numpy(self.cell.make_kpts(k_mesh)).to(device)

        self.nkpts = len(self.kpts)
        self.type = type
        nkx,nky,nkz = k_mesh

        Ls = np.array(list(product(range(nkx), range(nky), range(nkz))))
        Ls = np.dot(Ls,self.cell.a)*A2B/(np.pi*2)

        # Ls=cell.get_lattice_Ls(dimension=dimension,rcut=2*self.cell.rcut)

        self.Ls =torch.from_numpy(Ls).to(device)


    def real2k(self,hcore_R):
        hcore_k = torch.zeros((self.nkpts,self.cell.nao,self.cell.nao), dtype=self.type).to(device)

        for ik, kpt_frac in enumerate(self.kpts):
            for iR, R_int in enumerate(self.Ls):
                # !!! 修正之处 !!!
                # 同样，使用正确的坐标计算相位
                phase = torch.exp(1j * 2 * np.pi * torch.dot(kpt_frac, R_int)).to(device)
                hcore_k[ik] += phase * hcore_R[iR]
        hcore_k /= self.nkpts
        return hcore_k

    def k2real(self,hcore_k_direct):
        hcore_R_from_k = torch.zeros((len(self.Ls), self.cell.nao, self.cell.nao), dtype=self.type).to(device)

        for iR, R_int in enumerate(self.Ls):
            for ik, kpt_frac in enumerate(self.kpts):
                # !!! 修正之处 !!!
                # 使用分数坐标k_frac和整数坐标R_int计算正确的相位
                phase = torch.exp(-1j * 2 * np.pi * torch.dot(kpt_frac, R_int)).to(device)
                hcore_R_from_k[iR] += phase * hcore_k_direct[ik]


        return hcore_R_from_k.real


class post_processing_D(torch.nn.Module):
    def __init__(self,k_mesh: List):
        super().__init__()
        nkx, nky, nkz = k_mesh

        self.num = nkx*nky*nkz
    def forward(self, x):
        x=(x+x.transpose(-1,-2))/2

        out=[]
        step=x.shape[-1]//self.num
        for i in range(self.num):
            x_i=x[:step,i*step:(i+1)*step]

            out.append(x_i)

        return torch.stack(out)


class post_processing_X(torch.nn.Module):
    def __init__(self,k_mesh: List):
        super().__init__()
        nkx, nky, nkz = k_mesh
        self.num = nkx * nky * nkz

    def forward(self, x):
        x = (x - x.transpose(-1, -2)) / 2

        out = []
        step = x.shape[-1] // self.num
        for i in range(self.num):
            x_i = x[:step, i * step:(i + 1) * step]

            out.append(x_i)

        return torch.stack(out)


class BatchedConstrainedTraceLayer(torch.nn.Module):
    """
    一个自定义层，确保输出矩阵批次 B_out 中的每个矩阵都满足 Tr(A[i] * B_out[i]) = c。

    Args:
        A (torch.Tensor): 一个已知的、固定的三维张量 (batch_size, n, m)，代表一批矩阵。
        c (float): 目标迹值，对批次中每个矩阵都适用。
    """

    def __init__(self, A: torch.Tensor, c: float):
        super().__init__()
        if A.dim() != 3:
            raise ValueError("输入的矩阵 A 必须是一个三维张量 (batch_size, n, m)。")

        # 将 A 和 c 注册为 buffer
        self.register_buffer('A', A)
        self.register_buffer('c', torch.tensor(c, dtype=A.dtype))

        # 预先计算 A 中每个矩阵的 Frobenius 范数的平方 (Tr(A[i] * A[i]))
        # A * A 是逐元素乘积，形状为 (batch_size, n, m)
        # 我们需要在最后两个维度 (n, m) 上求和，保留 batch_size 维度
        # self.A_fro_sq 的形状将是 (batch_size,)
        self.register_buffer('A_fro_sq', torch.sum(A * A, dim=(-2, -1)))

        # 检查分母是否为零
        if torch.any(self.A_fro_sq == 0):
            print("警告: 矩阵 A 的批次中存在零矩阵，可能导致除以零错误。")

    def forward(self, B_raw: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            B_raw (torch.Tensor): 来自网络前一层的原始输出，一个三维张量 (batch_size, n, m)。
        Returns:
            torch.Tensor: 满足迹约束的最终输出矩阵批次 B_out。
        """
        if B_raw.shape != self.A.shape:
            raise ValueError(f"输入 B_raw 的形状 {B_raw.shape} 必须与 A 的形状 {self.A.shape} 匹配。")

        # 计算 Tr(A[i] * B_raw[i]) for each i in the batch
        # A * B_raw 的形状是 (batch_size, n, m)
        # 沿最后两个维度求和，得到每个矩阵的迹
        # trace_val 的形状是 (batch_size,)
        trace_val = torch.sum(self.A * B_raw, dim=(-2, -1))

        # 计算修正因子 alpha，现在 alpha 是一个向量
        # c (标量) - trace_val (向量) -> 广播
        # alpha 的形状是 (batch_size,)
        alpha = (self.c - trace_val) / self.A_fro_sq

        # 为了进行广播，需要将 alpha 的形状从 (batch_size,) 调整为 (batch_size, 1, 1)
        # 这样它就可以和形状为 (batch_size, n, m) 的 A 进行乘法
        # 修正项的形状是 (batch_size, n, m)
        correction = alpha.view(-1, 1, 1) * self.A

        # B_out 的形状是 (batch_size, n, m)
        B_out = B_raw + correction

        return B_out



class ConstrainedTraceLayer(torch.nn.Module):
    """
    一个自定义层，确保输出矩阵 B_out 满足 Tr(A * B_out) = c。

    Args:
        A (torch.Tensor): 已知的、固定的矩阵 A。
        c (float): 目标迹值。
    """

    def __init__(self, A: torch.Tensor, c: int):
        super().__init__()
        # 将 A 和 c 注册为 buffer，这样它们会被移动到正确的设备 (CPU/GPU)
        # 并且不会被视为模型参数进行训练。
        self.register_buffer('A', A)
        self.register_buffer('c', torch.tensor(c, dtype=A.dtype))

        # 预先计算 A 的转置和 Frobenius 范数的平方 (Tr(A * A^T))
        # 这个值是常数，可以避免重复计算。
        self.A_T = A.transpose(-2, -1).contiguous()
        # Tr(A * A^T) 等价于 A 中所有元素的平方和
        self.A_fro_sq = torch.sum(A * A)

    def forward(self, B_raw: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        Args:
            B_raw (torch.Tensor): 来自网络前一层的原始输出矩阵。
                                  可以是单个矩阵 (dim=2) 或一个批次 (dim=3)。
        Returns:
            torch.Tensor: 满足迹约束的最终输出矩阵 B_out。
        """
        # 为了处理批次数据 (batch)，我们需要适当调整维度用于广播
        # B_raw 的形状: (batch_size, rows, cols)
        # A 的形状: (rows, cols)

        # 计算 Tr(A * B_raw)
        # (A * B_raw) -> (batch_size, rows, cols), 逐元素相乘
        # torch.sum(..., dim=(-2, -1)) -> (batch_size), 在最后两个维度上求和

        # 计算修正因子 alpha
        # (c - trace_val) -> (batch_size)
        # alpha -> (batch_size)


        # 将 alpha 变形以匹配 B_raw 的维度进行广播
        # alpha.view(-1, 1, 1) -> (batch_size, 1, 1)
        # A_T 的形状: (cols, rows), 注意这里维度可能和 B_raw 不匹配，
        # B_out 和 B_raw 的维度应该一致，所以修正项的维度也要一致。
        # A_T 的形状应该是 (rows, cols) 的转置，即 (cols, rows)
        # 所以最终 B_out = B_raw + alpha * A^T 的维度会不匹配。

        # 修正公式：B_out = B_raw + alpha * A^T
        # B_raw: (batch, n, m)
        # A: (n, m) => A^T: (m, n)
        # 为了让维度匹配，修正项 A^T 需要和 B_raw 维度一致。
        # 这里 A 的维度应该和 B_raw 一致。

        # 重新检查公式： B_out = B_raw + correction
        # correction = alpha * A^T
        # B_raw 的维度是(N, M), A的维度是(N, M), A^T的维度是(M, N)
        # 这里存在一个问题，B_out的维度必须和B_raw一致。
        # Tr(A*B)中的A和B维度必须相同。
        # 所以A^T的维度也是(N,M)的转置，即(M,N)，这导致加法维度不匹配。

        # 正确的理解是：Tr(A * B) = <A^T, B> (Frobenius内积)
        # 校正方向应该是 A^T 的方向，但是 B_out 的维度需要保持不变。
        # 让我们回顾一下公式推导 Tr(A * (B_raw + alpha*X)) = c
        # Tr(A*B_raw) + alpha*Tr(A*X) = c => alpha = (c - Tr(A*B_raw)) / Tr(A*X)
        # 为了让 B_out 和 B_raw 维度一致，X 的维度必须和 B_raw 一致。
        # 最自然的选择是 X = A。
        # 此时 Tr(A*X) = Tr(A*A) = ||A||_F^2，和原来的分母一致。
        # 此时修正项为 alpha * A。
        #
        # 让我们用 X=A 重新验证:
        # B_out = B_raw + alpha * A
        # Tr(A * B_out) = Tr(A * (B_raw + alpha*A)) = Tr(A*B_raw) + alpha*Tr(A*A)
        # = Tr(A*B_raw) + (c - Tr(A*B_raw))/Tr(A*A) * Tr(A*A) = c
        # 这个是正确的。所以修正项应该是 alpha * A 而不是 alpha * A^T。

        # 重新实现
        trace_val = torch.sum(self.A * B_raw, dim=(-2, -1))
        alpha = (self.c - trace_val) / self.A_fro_sq

        # 修正项是 alpha * A
        correction = alpha.view(-1, 1, 1) * self.A.unsqueeze(0)

        B_out = B_raw + correction

        return B_out



class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = e3nn_nn.Extract(irreps_in, irreps_out_list, instructions)
        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = e3nn_nn.Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()

    def forward(self, x): return torch.cat(self.extr(x), dim=-1)

    def inverse(self, x): return torch.cat(self.extr_inv(x), dim=-1)


def irreps_from_l1l2(l1, l2, mul, no_parity=False):
    p = 1 if no_parity else (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = o3.Irreps([(mul, (l, p)) for l in required_ls])
    return required_irreps, required_irreps, None


import math


def merge_tensors_to_square_matrix(tensor_list):
    """
    将一个张量列表自动合并成一个正方形矩阵。

    函数会首先将列表中的所有张量展平并连接，然后计算元素总数。
    如果总数是一个完全平方数（如 9, 16, 25），它会自动推断出
    目标矩阵的维度（如 3x3, 4x4, 5x5）并返回结果。

    参数:
        tensor_list (list of torch.Tensor): 一个包含任意形状的 PyTorch 张量的列表。

    返回:
        torch.Tensor: 合并后的正方形矩阵。
        None: 如果元素总数不是一个完全平方数，无法构成正方形矩阵，则返回 None。
    """
    if not tensor_list:
        return torch.tensor([])  # 如果输入列表为空，返回一个空张量

    # 1. 将列表中的所有张量展平并连接成一个一维张量
    try:
        all_elements = torch.cat([t.flatten() for t in tensor_list])
    except Exception as e:
        print(f"处理张量时出错: {e}")
        return None

    # 2. 计算元素的总数量
    total_elements = all_elements.numel()

    if total_elements == 0:
        return torch.tensor([])  # 如果没有元素，返回一个空张量

    # 3. 计算边长并检查元素总数是否为完全平方数
    side_length = math.isqrt(total_elements)  # 使用整数开方，效率更高

    if side_length * side_length != total_elements:
        print(f"错误：元素总数 ({total_elements}) 不是一个完全平方数，无法构成正方形矩阵。")
        return None

    # 4. 将一维张量重塑为正方形矩阵
    matrix = all_elements.reshape(side_length, side_length)

    return matrix




def build_matrix_from_blocks(tensor_list: list, target_shape: tuple):
    """
    将一个张量列表作为“块”来构建一个目标矩阵，保留其二维方向性。

    此函数将 tensor_list 中的每个张量视为一个独立的块，并将其
    按顺序填充到目标矩阵中，类似于拼图或文字排版。

    参数:
        tensor_list (list of torch.Tensor): 包含作为构建块的 PyTorch 张量的列表。
        target_shape (tuple): 一个包含两个整数 (rows, cols) 的元组，
                                指定了目标矩阵的最终形状。

    返回:
        torch.Tensor: 构建完成的矩阵。
        None: 如果某个块无法放入目标矩阵中，则返回 None。
    """
    target_rows, target_cols = target_shape
    # 1. 创建画布
    target_matrix = torch.zeros(target_shape, dtype=default_dtype,device=device)

    # 2. 初始化光标和行高追踪器
    current_row, current_col = 0, 0
    row_max_height = 0

    for i, tensor in enumerate(tensor_list):
        # 确保所有张量都是二维的
        if tensor.dim() == 0: # e.g., tensor(0.5)
            tensor = tensor.reshape(1, 1)
        elif tensor.dim() == 1: # e.g., tensor([1, 2, 3])
            # 默认将一维张量视为行向量 (1, N)
            tensor = tensor.unsqueeze(0)

        h, w = tensor.shape

        # 3. 如果当前行放不下这个块，则换行
        #    (current_col > 0 是为了防止第一个块就直接换行)
        if current_col > 0 and current_col + w > target_cols:
            current_row += row_max_height
            current_col = 0
            row_max_height = 0

        # 4. 检查换行后，剩余空间是否还足够
        if current_row + h > target_rows or current_col + w > target_cols:
            print(f"错误: 第 {i} 个张量 (形状 {h}x{w}) 无法在 ({current_row}, {current_col}) "
                  f"位置放入目标矩阵 {target_shape} 中。空间不足。")
            return None

        # 5. 将张量块“嵌入”到目标矩阵中
        target_matrix[current_row : current_row + h, current_col : current_col + w] = tensor

        # 6. 更新光标和当前行的最大高度
        current_col += w
        row_max_height = max(row_max_height, h)

    return target_matrix

class e3TensorDecomp:
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch=default_dtype, no_parity=False, if_sort=True,
                 device_torch=device):
        self.dtype = default_dtype_torch
        self.device = device_torch
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = o3.Irreps(net_irreps_out)

        required_irreps_out_unsorted = o3.Irreps()
        in_slices = [0]
        wms = []
        H_slices = [0]
        wms_H = []

        for H_l1, H_l2 in out_js_list:
            mul = 1
            _, required_single_block_ir, _ = irreps_from_l1l2(H_l1, H_l2, mul, no_parity=no_parity)
            required_irreps_out_unsorted += required_single_block_ir

            in_slices.append(required_irreps_out_unsorted.dim)
            H_slices.append(H_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1))

            wm_block = []
            wm_H_block = []
            for _, ir_out_of_single_block in required_single_block_ir:
                cg = o3.wigner_3j(H_l1, H_l2, ir_out_of_single_block.l, dtype=default_dtype_torch, device=device_torch)
                wm_block.append(cg)
                cg_H = o3.wigner_3j(ir_out_of_single_block.l, H_l1, H_l2, dtype=default_dtype_torch,
                                    device=device_torch) * (2 * ir_out_of_single_block.l + 1)
                wm_H_block.append(cg_H)

            if wm_block:
                wms.append(torch.cat(wm_block, dim=-1))
            else:
                wms.append(torch.empty((2 * H_l1 + 1, 2 * H_l2 + 1, 0), dtype=default_dtype_torch, device=device_torch))
            if wm_H_block:
                wms_H.append(torch.cat(wm_H_block, dim=0))
            else:
                wms_H.append(
                    torch.empty((0, 2 * H_l1 + 1, 2 * H_l2 + 1), dtype=default_dtype_torch, device=device_torch))

        self.in_slices = in_slices
        self.wms = wms
        self.H_slices = H_slices
        self.wms_H = wms_H
        self.required_irreps_out_unsorted_simplified = required_irreps_out_unsorted.simplify()
        self.sort = None
        if if_sort:
            self.sort = sort_irreps(self.required_irreps_out_unsorted_simplified)
            self.required_irreps_out = self.sort.irreps_out
        else:
            self.required_irreps_out = self.required_irreps_out_unsorted_simplified
        if net_irreps_out is not None:
            if net_irreps_out != self.required_irreps_out:
                raise AssertionError(
                    f'Provided net_irreps_out {net_irreps_out} does not match e3TensorDecomp internal target {self.required_irreps_out}. Unsorted target was {self.required_irreps_out_unsorted_simplified}.')

    def get_H(self, net_out):
        if self.sort is not None:
            net_out_processed = self.sort.inverse(net_out)
        else:
            net_out_processed = net_out
        out_H_blocks = []
        for i in range(len(self.out_js_list)):
            block_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block_segment = net_out_processed.narrow(-1, block_slice.start,
                                                             block_slice.stop - block_slice.start)
            current_wm = self.wms[i]
            if net_out_block_segment.shape[-1] == 0:
                l1, l2 = self.out_js_list[i]
                num_elements_in_H_block = (2 * l1 + 1) * (2 * l2 + 1)
                H_block = torch.zeros(net_out.shape[0], num_elements_in_H_block, dtype=self.dtype, device=self.device)
            else:
                H_block = torch.einsum("ijk,bk->bij", current_wm, net_out_block_segment)
            #out_H_blocks.append(H_block.reshape(net_out.shape[0], -1))
        #return torch.cat(out_H_blocks, dim=-1)
            out_H_blocks.append(H_block.squeeze(0))
        return out_H_blocks

    def get_net_out(self, H_flat_batch):
        out_net_segments = []
        for i in range(len(self.out_js_list)):
            H_slice = slice(self.H_slices[i], self.H_slices[i + 1])
            l1, l2 = self.out_js_list[i]
            H_block_flat_segment = H_flat_batch.narrow(-1, H_slice.start, H_slice.stop - H_slice.start)
            H_block_reshaped = H_block_flat_segment.reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
            current_wm_H = self.wms_H[i]
            if current_wm_H.shape[0] == 0:
                net_segment = torch.empty(H_block_reshaped.shape[0], 0, dtype=self.dtype, device=self.device)
            else:
                net_segment = torch.einsum("kij,bij->bk", current_wm_H, H_block_reshaped)
            out_net_segments.append(net_segment)
        concatenated_net_out = torch.cat(out_net_segments, dim=-1)
        if self.sort is not None:
            concatenated_net_out = self.sort(concatenated_net_out)
        return concatenated_net_out




class ShiftedSoftPlus(torch.nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x): return self.softplus(x) - self._log2


class EquivariantLayerNormFast(torch.nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
            self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        '''
            Use torch layer norm for scalar features.
        '''

        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw:(iw + mul)]
                bias = self.affine_bias[ib:(ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)
        return output


# class EquivariantLayerNormFast(torch.nn.Module):
#     def __init__(self, irreps, eps=1e-20, affine=True, normalization='component'):
#         super().__init__()
#         self.irreps = o3.Irreps(irreps)
#         self.eps = eps
#         self.affine = affine
#         num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
#         num_features = self.irreps.num_irreps
#         if affine:
#             self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
#             self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
#         else:
#             self.register_parameter('affine_weight', None)
#             self.register_parameter('affine_bias', None)
#         assert normalization in ['norm', 'component']
#         self.normalization = normalization
#
#     def forward(self, node_input, **kwargs):
#         dim = node_input.shape[-1]
#         fields = []
#         ix = 0;
#         iw = 0;
#         ib = 0
#         for mul, ir in self.irreps:
#             d = ir.dim
#             field = node_input.narrow(1, ix, mul * d)
#             ix += mul * d
#             if ir.l == 0 and ir.p == 1:
#                 if self.affine:
#                     weight = self.affine_weight[iw:(iw + mul)]
#                     bias = self.affine_bias[ib:(ib + mul)]
#                     field = torch.nn.functional.layer_norm(field, (mul * d,),
#                                                            weight.repeat_interleave(d) if d > 1 else weight,
#                                                            bias.repeat_interleave(d) if d > 1 else bias, self.eps)
#                 else:
#                     field = torch.nn.functional.layer_norm(field, (mul * d,), None, None, self.eps)
#                 fields.append(field.reshape(-1, mul * d))
#                 iw += mul;
#                 ib += mul
#                 continue
#             field = field.reshape(-1, mul, d)
#             if self.normalization == 'norm':
#                 field_norm = field.pow(2).sum(-1)
#             elif self.normalization == 'component':
#                 field_norm = field.pow(2).mean(-1)
#             else:
#                 raise ValueError(f"Invalid normalization: {self.normalization}")
#             field_norm = torch.mean(field_norm, dim=1, keepdim=True)
#             field_norm = (field_norm + self.eps).rsqrt()
#             if self.affine:
#                 weight = self.affine_weight[None, iw:(iw + mul)]
#                 field_norm = field_norm * weight
#             iw += mul
#             field = field * field_norm.reshape(-1, mul, 1)
#             fields.append(field.reshape(-1, mul * d))
#         assert ix == dim, f"{ix} != {dim}"
#         return torch.cat(fields, dim=-1)


@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = o3.Irreps(irreps_head)
        self.irreps_mid_in = o3.Irreps([(mul * num_heads, ir) for mul, ir in self.irreps_head])
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx += mul * ir.dim

    def forward(self, x):
        N, _ = x.shape;



        # Re-implementing the original forward logic correctly:
        out_list_reshaped_segments = []
        for _, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            segment = x.narrow(1, start_idx, end_idx - start_idx)  # [N, mul_mid_in * dim_ir_mid_in]
            # mul_mid_in = mul_head * num_heads. dim_ir_mid_in = dim_ir_head
            # We want to reshape to [N, num_heads, mul_head * dim_ir_head]
            # The -1 in reshape will be mul_head * dim_ir_head
            reshaped_segment = segment.reshape(x.shape[0], self.num_heads, -1)
            out_list_reshaped_segments.append(reshaped_segment)

        # Concatenate these segments along the last dimension (the feature dimension within a head)
        return torch.cat(out_list_reshaped_segments, dim=2)


@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    def __init__(self, irreps_head_single_definition: o3.Irreps):
        super().__init__()
        self.irreps_head_single = o3.Irreps(irreps_head_single_definition)


        self.output_irreps_structure = []  # To guide the final cat
        temp_start_dim_in_head = 0
        for mul_h, ir_h in self.irreps_head_single:
            dim_h_ir = mul_h * ir_h.dim
            self.output_irreps_structure.append({
                "slice_in_head": slice(temp_start_dim_in_head, temp_start_dim_in_head + dim_h_ir),
                "output_mul": mul_h  # Multiplicity for this irrep in the final output (will be scaled by num_heads)
            })
            temp_start_dim_in_head += dim_h_ir

    def forward(self, x_heads: torch.Tensor):  # x_heads is [N, num_heads, total_dim_single_head]
        N, num_h, _ = x_heads.shape
        output_segments_by_ir_type = []
        for ir_info in self.output_irreps_structure:
            # segment_for_ir_type_all_heads is [N, num_heads, dim_of_this_ir_in_head]
            segment_for_ir_type_all_heads = x_heads[:, :, ir_info["slice_in_head"]]

            reshaped_segment = segment_for_ir_type_all_heads.transpose(1, 2).reshape(N, -1)
            output_segments_by_ir_type.append(reshaped_segment)
        return torch.cat(output_segments_by_ir_type, dim=1)


def find_positions_in_tensor_fast(tensor):  # Duplicated, ensure one definition
    unique_elements, inverse_indices = torch.unique(tensor, sorted=True, return_inverse=True)
    positions = {element.item(): torch.nonzero(inverse_indices == i, as_tuple=True)[0] for i, element in
                 enumerate(unique_elements)}
    return positions


# find_positions_in_tensor_fast=torch.compile(find_positions_in_tensor_fast) # Compile outside class

#from matten
class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        node_attr:o3.Irreps,
        internal_and_share_weights: bool = False,
    ):
        """
        UVU tensor product.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.

        """

        super().__init__()

        self.out=irreps_out
        self.node_attr=node_attr

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == o3.Irreps("0e"):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort irreps_mid to let irreps of the same type be adjacent to each other
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.lin0=o3.FullyConnectedTensorProduct(irreps_in1, self.node_attr,irreps_in1)

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )

        # self.lin=o3.Linear(irreps_in=self.irreps_mid,irreps_out=self.out)
        self.lin=o3.FullyConnectedTensorProduct(self.irreps_mid, self.node_attr,self.out)

        self.sc = o3.FullyConnectedTensorProduct(
            irreps_in1, self.node_attr, self.out
        )


    def forward( self, data1: torch.tensor, data2: torch.tensor, data_weight: torch.tensor,data3:torch.tensor
    ) -> torch.tensor:
        node_feats = data1
        node_attrs = data3
        edge_attrs = data2

        # node_sc = self.sc(node_feats, node_attrs)


        node_feats = self.lin0(node_feats, node_attrs)

        node_feats = self.tp(node_feats, edge_attrs, data_weight)


        # update
        node_conv_out = self.lin(node_feats, node_attrs)

        # node_feats = node_sc + node_conv_out
        node_feats=node_conv_out
        return node_feats

from torch_scatter import scatter

class Attention(torch_nn.Module):
    def __init__(self, node_attr_irreps, irreps_node_input, irreps_query, irreps_key, irreps_value, num_radial_basis):
        super().__init__()
        self.heads = heads
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)

        self.num_radial_basis = num_radial_basis
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.node_attr_irreps = o3.Irreps(node_attr_irreps)

        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_value = o3.Irreps(irreps_value)

        self.h_q = o3.FullyConnectedTensorProduct(self.irreps_node_input, self.node_attr_irreps, self.irreps_query)

        self.tp_k = UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_key, self.node_attr_irreps)

        self.fc_k = e3nn_nn.FullyConnectedNet(
            [self.num_radial_basis] + [100] + [self.tp_k.tp.weight_numel if self.tp_k.tp else 0],
            act=torch.nn.functional.silu)
        self.tp_v = UVUTensorProduct(self.irreps_node_input, self.irreps_sh, self.irreps_value, self.node_attr_irreps)
        self.fc_v = e3nn_nn.FullyConnectedNet(
            [self.num_radial_basis] + [100] + [self.tp_v.tp.weight_numel if self.tp_v.tp else 0],
            act=torch.nn.functional.silu)

        self.irreps_query_head = o3.Irreps([(mul // self.heads, ir) for mul, ir in self.irreps_query if
                                            mul % self.heads == 0 and mul // self.heads > 0]).simplify()
        self.irreps_key_head = o3.Irreps([(mul // self.heads, ir) for mul, ir in self.irreps_key if
                                          mul % self.heads == 0 and mul // self.heads > 0]).simplify()
        self.irreps_value_head = o3.Irreps([(mul // self.heads, ir) for mul, ir in self.irreps_value if
                                            mul % self.heads == 0 and mul // self.heads > 0]).simplify()

        if not self.irreps_query_head: self.irreps_query_head = o3.Irreps("0x0e")  # Handle empty case
        if not self.irreps_key_head: self.irreps_key_head = o3.Irreps("0x0e")
        if not self.irreps_value_head: self.irreps_value_head = o3.Irreps("0x0e")

        self.vec2headsq = Vec2AttnHeads(self.irreps_query_head, self.heads)
        self.vec2headsk = Vec2AttnHeads(self.irreps_key_head, self.heads)
        self.vec2headsv = Vec2AttnHeads(self.irreps_value_head, self.heads)

        self.dot_prods = torch_nn.ModuleList(
            [o3.FullyConnectedTensorProduct(self.irreps_query_head, self.irreps_key_head, "0e") for _ in
             range(self.heads)])

        self.heads2vecv = AttnHeads2Vec(self.irreps_value_head)
        self.final_output_irreps = o3.Irreps([(mul * self.heads, ir) for mul, ir in self.irreps_value_head]).simplify()

        if not self.final_output_irreps: self.final_output_irreps = o3.Irreps("0x0e")

        self.lin_out = o3.FullyConnectedTensorProduct(self.final_output_irreps, self.node_attr_irreps,
                                                      self.final_output_irreps)

        self.norm = EquivariantLayerNormFast(irreps=self.final_output_irreps)

        self.dim_sqrt = max(1.0, self.irreps_key_head.dim) ** 0.5  # Avoid div by zero if key_head is empty
        self.sc = o3.FullyConnectedTensorProduct(
            irreps_node_input, node_attr_irreps, self.final_output_irreps
        )
    def forward(self, node_attr_batch, node_input_batch, edge_src, edge_dst, edge_sh_attr, edge_radial_scalars,
                edge_length_cutoff_factor, fpit_edge_dst) -> torch.Tensor:

        q_node = self.h_q(node_input_batch, node_attr_batch)
        k_weights = self.fc_k(edge_radial_scalars)

        k_edge = self.tp_k(node_input_batch[edge_src], edge_sh_attr, k_weights, node_attr_batch[edge_src])
        node_input_sc = self.sc(node_input_batch, node_attr_batch)
        v_weights = self.fc_v(edge_radial_scalars)
        v_edge = self.tp_v(node_input_batch[edge_src], edge_sh_attr, v_weights, node_attr_batch[edge_src])
        if heads==1:
            q_heads=q_node
            k_heads=k_edge
            v_heads=v_edge
        else:
            q_heads = self.vec2headsq(q_node)

            k_heads = self.vec2headsk(k_edge)

            v_heads = self.vec2headsv(v_edge)
        q_heads_dst = q_heads[edge_dst]

        alpha_exp_sum_heads = []
        for head_idx in range(self.heads):
            q_h = q_heads_dst.narrow(1, head_idx, 1).squeeze(
                1) if self.heads > 1 else q_heads_dst  # More robust slicing for heads
            k_h = k_heads.narrow(1, head_idx, 1).squeeze(1) if self.heads > 1 else k_heads
            v_h = v_heads.narrow(1, head_idx, 1).squeeze(1) if self.heads > 1 else v_heads
            if q_h.shape[-1] == 0 or k_h.shape[-1] == 0:  # Handle empty irreps for dot product
                dot = torch.zeros(q_h.shape[0], 1, device=q_h.device, dtype=q_h.dtype)  # scalar output
            else:
                dot = self.dot_prods[head_idx](q_h, k_h) / self.dim_sqrt

            attn_scores_softmax = torch.zeros_like(dot)
            for node_idx_key in fpit_edge_dst:
                edge_indices = fpit_edge_dst[node_idx_key]
                if len(edge_indices) > 0:  # Ensure there are scores to softmax
                    attn_scores_softmax[edge_indices] = torch.nn.functional.softmax(dot[edge_indices], dim=0)
            alpha = attn_scores_softmax * edge_length_cutoff_factor[:, None]

            weighted_v = alpha * v_h

            output_head_summed = scatter(weighted_v, edge_dst, dim=0, dim_size=node_input_batch.shape[0], reduce='sum')


            alpha_exp_sum_heads.append(output_head_summed)

        if not alpha_exp_sum_heads:  # If heads = 0 or no contributions
            return torch.zeros(node_input_batch.shape[0], self.final_output_irreps.dim, device=node_input_batch.device,
                               dtype=node_input_batch.dtype)
        aggregated_heads = torch.stack(alpha_exp_sum_heads, dim=1)

        combined_v = self.heads2vecv(aggregated_heads)
        output_final = self.lin_out(combined_v, node_attr_batch)
        output_final = output_final+node_input_sc
        # output_normed = self.norm(output_final)

        return output_final

act = {
    1: torch.nn.functional.silu,
    -1: torch.tanh,
}
act_gates = {
    1: ShiftedSoftPlus(),
    -1: torch.tanh,
}

from typing import Optional
from torch_sparse import SparseTensor

class EdgeFeaturesFromNodes(torch.nn.Module):
    """
    一个通过节点特征的张量积来生成边特征的模块。

    这个类会为图中的每一条边，获取其源节点和目标节点的特征，
    然后计算这两个节点特征的张量积，从而得到该边的特征。
    """
    def __init__(self,
                 irreps_node_in: o3.Irreps,
                 irreps_edge_out: Optional[o3.Irreps] = None,
                 num_radial_basis: int = 10):
        """
        初始化方法。

        Args:
            irreps_node_in (o3.Irreps): 输入的节点特征的不可约表示 (Irreps)。
            irreps_edge_out (o3.Irreps, optional): 期望的输出边特征的Irreps。
                                                   如果为 None，则输出完整的张量积结果。
                                                   如果提供，则会将完整的张量积结果通过一个线性层投影到这个指定的Irreps。
                                                   默认为 None。
        """
        super().__init__()
        self.irreps_node_in = o3.Irreps(irreps_node_in)

        self.irreps_edge_out = o3.Irreps(irreps_edge_out)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)

        self.tp_edge = o3.FullyConnectedTensorProduct(self.irreps_node_in, self.irreps_sh,
                                                      o3.FullTensorProduct(self.irreps_sh, self.irreps_node_in).irreps_out.simplify())

        # self.fc = e3nn_nn.FullyConnectedNet([num_radial_basis]  + [100] + [self.tp.weight_numel if self.tp else 0],act=torch.nn.functional.silu)

        #    定义完整的张量积操作
        #    它会计算两个节点特征的所有可能的乘积项。
        self.tp = o3.FullyConnectedTensorProduct(self.tp_edge.irreps_out, self.irreps_node_in,
                                                 self.irreps_edge_out, shared_weights=False)

        self.fc = e3nn_nn.FullyConnectedNet([num_radial_basis] + [100] + [self.tp.weight_numel if self.tp else 0],
                                            act=torch.nn.functional.silu)

        # 张量积操作的完整输出Irreps
        self.irreps_tp_out = self.tp.irreps_out

        # 如果指定了输出Irreps，则创建一个线性投影层
        if irreps_edge_out is not None:
            self.irreps_edge_out = o3.Irreps(irreps_edge_out)
            self.linear = o3.Linear(self.irreps_tp_out, self.irreps_edge_out)
        else:
            # 如果未指定，输出就是完整的张量积结果
            self.irreps_edge_out = self.irreps_tp_out
            self.linear = None
    def forward(self, node_features: torch.Tensor, edge_sh:torch.Tensor, edge_length_embedded:torch.Tensor,
                edge_src: torch.Tensor,edge_dst:torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            node_features (torch.Tensor): 包含所有节点特征的张量，形状为 [num_nodes, irreps_node_in.dim]。
            edge_index (torch.Tensor): 图的边索引，形状为 [2, num_edges]。

        Returns:
            torch.Tensor: 生成的边特征张量，形状为 [num_edges, irreps_edge_out.dim]。
        """
        # 根据边索引，获取每条边的源节点和目标节点的特征
        src_features = node_features[edge_src]
        dst_features = node_features[edge_dst]

        edge_sh_attr = o3.spherical_harmonics(self.irreps_sh, edge_sh, True, normalization="component")


        # 计算源节点和目标节点特征的张量积
        edge_features_tp = self.tp_edge(dst_features,edge_sh_attr)


        edge_features_out = self.tp(edge_features_tp,src_features, self.fc(edge_length_embedded))
        adj = SparseTensor(row=edge_src, col=edge_dst, value=edge_features_out,
                           sparse_sizes=(int(edge_src.max())+1, int(edge_src.max())+1))
        adj_coalesced = adj.coalesce(reduce='add')
        row, col, val = adj_coalesced.coo()

        # # 如果定义了线性投影层，则进行投影
        # if self.linear:
        #     edge_features_out = self.linear(edge_features_tp)
        # else:
        #     edge_features_out = edge_features_tp
        return row, col, val

#from DeepH-E3
# class EdgeUpdateBlock(torch.nn.Module):
#     def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
#                  act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm',
#                  if_sort_irreps=False):
#         super(EdgeUpdateBlock, self).__init__()
#         irreps_in_node = Irreps(irreps_in_node)
#         irreps_in_edge = Irreps(irreps_in_edge)
#         irreps_out_edge = Irreps(irreps_out_edge)
#
#         irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
#         if if_sort_irreps:
#             self.sort = sort_irreps(irreps_in1)
#             irreps_in1 = self.sort.irreps_out
#         irreps_in2 = irreps_sh
#
#         self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
#
#         self.nonlin = None
#         self.lin_post = None
#         if nonlin:
#             self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
#             irreps_conv_out = self.nonlin.irreps_in
#             conv_nonlin = False
#         else:
#             irreps_conv_out = irreps_out_edge
#             conv_nonlin = True
#
#         self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act,
#                              act_gates=act_gates)
#         self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
#
#         if use_sc:
#             self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species ** 2}x0e', self.conv.irreps_out)
#
#         if nonlin:
#             self.irreps_out = self.nonlin.irreps_out
#         else:
#             self.irreps_out = self.conv.irreps_out
#
#         self.norm = None
#         if norm:
#             if norm == 'e3LayerNorm':
#                 self.norm = e3LayerNorm(self.irreps_out)
#             else:
#                 raise ValueError(f'unknown norm: {norm}')
#
#         self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out)  # ! consider init_edge
#
#         self.self_tp = None
#         if use_selftp:
#             self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
#
#         self.use_sc = use_sc
#         self.init_edge = init_edge
#         self.if_sort_irreps = if_sort_irreps
#         self.irreps_in_edge = irreps_in_edge
#
#     def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
#
#         if not self.init_edge:
#             edge_fea_old = edge_fea
#             if self.use_sc:
#                 edge_self_connection = self.sc(edge_fea, edge_one_hot)
#             edge_fea = self.lin_pre(edge_fea)
#
#         index_i = edge_index[0]
#         index_j = edge_index[1]
#         fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
#         if self.if_sort_irreps:
#             fea_in = self.sort(fea_in)
#         edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
#
#         edge_fea = self.lin_post(edge_fea)
#
#         if self.use_sc:
#             edge_fea = edge_fea + edge_self_connection
#
#         if self.nonlin is not None:
#             edge_fea = self.nonlin(edge_fea)
#
#         if self.norm is not None:
#             edge_fea = self.norm(edge_fea, batch[edge_index[0]])
#
#         if not self.init_edge:
#             edge_fea = self.skip_connect(edge_fea_old, edge_fea)
#
#         if self.self_tp is not None:
#             edge_fea = self.self_tp(edge_fea)
#
#         return edge_fea

class CoreGNN(torch_nn.Module):
    def __init__(self, irreps_node_attr_in: o3.Irreps, irreps_node_embed_in: o3.Irreps,
                 irreps_query: o3.Irreps, irreps_key: o3.Irreps,
                 irreps_node_hidden: o3.Irreps, irreps_node_output: o3.Irreps,
                 num_attn_layers: int, num_radial_basis: int):
        super().__init__()

        self.irreps_node_attr_in = o3.Irreps(irreps_node_attr_in)
        self.irreps_node_embed_in = o3.Irreps(irreps_node_embed_in)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.layers = torch_nn.ModuleList()

        current_irreps_node = self.irreps_node_embed_in
        for _ in range(num_attn_layers):
            attn_value_irreps = irreps_node_hidden

            irreps_scalars_gate_in = o3.Irreps([(m, i) for m, i in attn_value_irreps if i.l == 0]).simplify()
            irreps_gated_gate_in = o3.Irreps([(m, i) for m, i in attn_value_irreps if i.l > 0]).simplify()
            irreps_gates_for_gate = o3.Irreps([(m, (0, 1)) for m, _ in irreps_gated_gate_in]).simplify()  # p=1 for 0e

            gate = e3nn_nn.Gate(
                irreps_scalars_gate_in,
                [act[ir.p] for _, ir in irreps_scalars_gate_in],  # scalar
                irreps_gates_for_gate,
                [act_gates[ir.p] for _, ir in irreps_gates_for_gate],  # gates (scalars)
                irreps_gated_gate_in,  # gated tensors
            )

            attn_layer = Attention(
                node_attr_irreps=self.irreps_node_attr_in,
                irreps_node_input=current_irreps_node,
                irreps_query=irreps_query, irreps_key=irreps_key, irreps_value=gate.irreps_in,
                num_radial_basis=num_radial_basis
            )




            norm_layer = EquivariantLayerNormFast(gate.irreps_out).to(device)
            self.layers.append(torch_nn.ModuleDict({'attn': attn_layer, 'gate': gate, 'norm': norm_layer}))
            current_irreps_node = gate.irreps_out
        self.layers_f=Attention(self.irreps_node_attr_in,
                                        current_irreps_node, irreps_query, irreps_key,
                                         self.irreps_node_output, num_radial_basis)




        self.final_node_proj = o3.Linear(self.irreps_node_output, self.irreps_node_output,biases=True)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)


    def forward(self, node_attr_batch, node_features_batch, edge_src, edge_dst, edge_vec_batch,
                edge_radial_scalars_batch, max_radius_val):

        edge_sh_attr = o3.spherical_harmonics(self.irreps_sh, edge_vec_batch, True, normalization="component")
        edge_length = edge_vec_batch.norm(dim=1)
        edge_length_cutoff_factor = e3nn.math.soft_unit_step(10 * (1 - edge_length / max_radius_val))
        fpit_edge_dst = find_positions_in_tensor_fast(edge_dst)
        current_node_features = node_features_batch
        for layer_module_dict in self.layers:
            attn_out = layer_module_dict['attn'](node_attr_batch, current_node_features, edge_src, edge_dst,
                                                 edge_sh_attr, edge_radial_scalars_batch, edge_length_cutoff_factor,
                                                 fpit_edge_dst)

            gated_out = layer_module_dict['gate'](attn_out)
            # print(gated_out)
            #print(layer_module_dict['gate'].irreps_out,gated_out)
            current_node_features = layer_module_dict['norm'](gated_out)
            # current_node_features = gated_out
            # print(current_node_features)

            #current_node_features=gated_out
        current_node_features = self.layers_f(node_attr_batch,current_node_features, edge_src, edge_dst, edge_sh_attr, edge_radial_scalars_batch, edge_length_cutoff_factor, fpit_edge_dst)
        final_node_features = self.final_node_proj(current_node_features)

        return final_node_features



class DensityPredictingGNN(torch_nn.Module):
    def __init__(self, irreps_node_attr_in_str: str, irreps_node_embed_in_str: str,
                 irreps_query_str: str, irreps_key_str: str, irreps_node_hidden_str: str,
                 irreps_core_gnn_node_out_str: str, num_attn_layers: int, num_radial_basis: int,atom_to_basis_map: Dict[str, str]):
        super().__init__()
        self.atom_embedding = o3.Linear(o3.Irreps(irreps_node_attr_in_str),
                                        o3.Irreps(irreps_node_embed_in_str), internal_weights=True).to(device)
        self.core_gnn = CoreGNN(
            irreps_node_attr_in=o3.Irreps(irreps_node_attr_in_str),
            irreps_node_embed_in=o3.Irreps(irreps_node_embed_in_str),
            irreps_query=o3.Irreps(irreps_query_str), irreps_key=o3.Irreps(irreps_key_str),
            irreps_node_hidden=o3.Irreps(irreps_node_hidden_str),
            irreps_node_output=o3.Irreps(irreps_core_gnn_node_out_str),
            num_attn_layers=num_attn_layers, num_radial_basis=num_radial_basis
        )

        self.egde_feature_builder = EdgeFeaturesFromNodes(
            irreps_node_in=o3.Irreps(irreps_core_gnn_node_out_str),
            irreps_edge_out=o3.Irreps(irreps_core_gnn_node_out_str),num_radial_basis=num_radial_basis,
        )

        self.density_matrix_feature_builder = DensityMatrixFeatureBuilder(
            irreps_node_superset=o3.Irreps(irreps_core_gnn_node_out_str),num_radial_basis=num_radial_basis,
            atom_to_basis_irreps=atom_to_basis_map,node_attr=o3.Irreps(irreps_node_attr_in_str)
        )
        self.atom_to_basis_irreps = {k: o3.Irreps(v) for k, v in atom_to_basis_map.items()}
        self.max_radius_for_gnn = radius
        self.num_radial_basis = num_radial_basis

        self.register_buffer('_cached_batch', None)
        self.register_buffer('_cached_node_attr_input', None)
        self.register_buffer('_cached_edge_src', None)
        self.register_buffer('_cached_edge_dst', None)
        self.register_buffer('_cached_edge_vec', None)

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=default_dtype)

        edge_src = data['edge_index'][0][0]  # Edge source
        edge_dst = data['edge_index'][0][1]  # Edge destination

        # We need to compute this in the computation graph to backprop to positions
        # We are computing the relative distances + unit cell shifts from periodic boundaries
        edge_batch = batch[edge_src]

        edge_vec = (data['pos'][edge_dst]
                    - data['pos'][edge_src]
                    + torch.einsum('ni,nij->nj', data['edge_shift'][0], data['lattice'][edge_batch]))
        print('test')
        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self, data_batch: Data) -> torch.Tensor:
        # 步骤2: 检查缓存是否已存在 (只需检查一个即可)
        if self._cached_batch is not None:
            # 如果缓存存在，直接从所有缓冲区加载结果
            # print("--- Using globally cached preprocess results ---") # 用于调试
            batch = self._cached_batch
            node_attr_input = self._cached_node_attr_input
            edge_src = self._cached_edge_src
            edge_dst = self._cached_edge_dst
            edge_vec = self._cached_edge_vec
        else:
            # 如果缓存不存在，说明是第一次运行
            # print("--- Running preprocess for the first and only time ---") # 用于调试

            # 步骤3: 执行一次预处理
            processed_results = self.preprocess(data_batch)
            b, n_attr, e_src, e_dst, e_vec = processed_results

            # 步骤4: 将结果分别存入对应的缓冲区
            # 使用 .clone().detach() 是确保安全的最佳实践
            self._cached_batch = b.clone().detach()
            self._cached_node_attr_input = n_attr.clone().detach()
            self._cached_edge_src = e_src.clone().detach()
            self._cached_edge_dst = e_dst.clone().detach()
            self._cached_edge_vec = e_vec.clone().detach()

            # 将新计算的结果赋给局部变量，以供后续使用
            batch, node_attr_input, edge_src, edge_dst, edge_vec = \
                self._cached_batch, self._cached_node_attr_input, self._cached_edge_src, self._cached_edge_dst, self._cached_edge_vec

        # --- 从这里开始，后续计算在每次调用forward时都会执行 ---
        # --- 它们使用的是从上方代码块获得的（新计算或缓存的）变量 ---

        node_features_embedded = self.atom_embedding(node_attr_input)

        edge_length_gnn = edge_vec.norm(dim=1)
        edge_radial_scalars_gnn = e3nn.math.soft_one_hot_linspace(
            edge_length_gnn, start=0.0, end=self.max_radius_for_gnn,
            number=self.num_radial_basis,
            basis="smooth_finite", cutoff=True,
        ).mul(self.num_radial_basis ** 0.5)

        node_atom_features_batch = self.core_gnn(
            node_attr_input, node_features_embedded,
            edge_src,  # 使用从缓存中获取的edge_src
            edge_dst,  # 使用从缓存中获取的edge_dst
            edge_vec, edge_radial_scalars_gnn, self.max_radius_for_gnn
        )

        row,col,edge_features = self.egde_feature_builder(node_atom_features_batch,edge_vec,
                                                  edge_radial_scalars_gnn, edge_src, edge_dst)


        dm = self.density_matrix_feature_builder(
            node_atom_features_batch,edge_features,torch.stack([row,col]),
            data_batch, edge_radial_scalars_gnn, node_attr_input
            # node_attr_input, node_features_embedded,
            # edge_src,  # 使用从缓存中获取的edge_src
            # edge_dst,  # 使用从缓存中获取的edge_dst
            # edge_vec, edge_radial_scalars_gnn, self.max_radius_for_gnn
        )

        return dm


unpaired_electrons = {
    'H': 1,   # H
    'He': 0,   # He
    'Li': 1,   # Li
    'Be': 0,   # Be
    'B': 1,   # B
    'C': 2,   # C
    'N': 3,   # N
    'O': 2,   # O
    'F': 1,   # F
    'Ne': 0,  # Ne
    'Na': 1,  # Na
    'Mg': 0,  # Mg
    'Al': 1,  # Al
    'Si': 2,  # Si
    'P': 3,  # P
    'S': 2,  # S
    'Cl': 1,  # Cl
    'Ar': 0,  # Ar
    # 可以继续添加更多元素
}
def out_js(symbol_j,symbol_k):
    from pyscf import gto
    j=gto.M(f"{symbol_j} 0 0 0",basis=basis,spin=unpaired_electrons[symbol_j], verbose=0)
    k=gto.M(f"{symbol_k} 0 0 0",basis=basis,spin=unpaired_electrons[symbol_k],verbose=0)
    lis_j = [i.split()[2][1] for i in j.ao_labels()]
    lis_k = [i.split()[2][1] for i in k.ao_labels()]
    out_js_list= []

    n=0
    lj=[]
    lk=[]
    for i in lis_j:
        if i=='s':
            lj.append(0)
        else:
            n+=1
            if n%3==0:
                lj.append(1)
    n=0
    for i in lis_k:
        if i=='s':
            lk.append(0)
        else:
            n+=1
            if n%3==0:
                lk.append(1)

    for i in lj:
        for j in lk:
            out_js_list.append((i,j))
    return out_js_list

def net_output(symbol_i):
    from pyscf import gto
    i=gto.M(f"{symbol_i} 0 0 0",basis=basis,spin=unpaired_electrons[symbol_i], verbose=0)
    lis_i = [x.split()[2][1] for x in i.ao_labels()]
    n = 0
    li=''
    for i in lis_i:
        if i == 's':
            if li=='':
                li='0e'
            else:
                li+='+0e'
        else:
            n += 1
            if n % 3==0:
                if li=='':
                    li='1o'
                else:
                    li+='+1o'
    return {f'{symbol_i}':o3.Irreps(li).simplify()}

# class DensityMatrixFeatureBuilder(torch.nn.Module):
#     """
#     最终优化版：预先实例化所有可能的张量积模块，以提升性能并遵循最佳实践。
#     """
#
#     def __init__(self,
#                  irreps_node_superset: o3.Irreps,num_radial_basis: int,
#                  atom_to_basis_irreps: Dict[str, o3.Irreps],node_attr: o3.Irreps):
#         super().__init__()
#         self.node_attr=node_attr
#
#         self.irreps_node_superset = o3.Irreps(irreps_node_superset)
#
#         irreps_in1 = irreps_node_superset+irreps_node_superset
#
#         self.irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)
#
#         self.sort = sort_irreps(irreps_in1)
#
#         self.atom_to_basis_irreps = {k: o3.Irreps(v) for k, v in atom_to_basis_irreps.items()}
#         self.tp0=o3.FullTensorProduct(irreps_node_superset,irreps_node_superset)
#
#         self.tp_node=o3.FullyConnectedTensorProduct(irreps_node_superset,irreps_node_superset,self.tp0.irreps_out.simplify(),shared_weights=True)
#         self.tp_edge=o3.FullyConnectedTensorProduct(self.sort.irreps_out,self.irreps_sh,self.tp0.irreps_out.simplify(),shared_weights=False)
#
#         self.fc=e3nn_nn.FullyConnectedNet([num_radial_basis]  + [100] + [self.tp_edge.weight_numel if self.tp_edge else 0],act=torch.nn.functional.silu)
#         self.lin0 = o3.FullyConnectedTensorProduct(self.tp0.irreps_out.simplify(), self.node_attr, self.tp0.irreps_out.simplify())
#
#         self.out=None
#
#     def reduce(self,num,atom_symbols_sample):
#         out=dict()
#         out1=dict()
#         et=dict()
#         out_dim=dict()
#         for j in range(num):
#             for k in range(num):
#                 symbol_j = atom_symbols_sample[j]
#                 symbol_k = atom_symbols_sample[k]
#                 tpout=o3.FullTensorProduct(self.atom_to_basis_irreps[symbol_j],self.atom_to_basis_irreps[symbol_k]).irreps_out.simplify()
#                 out_dim.update({f'{symbol_j}_{symbol_k}': [self.atom_to_basis_irreps[symbol_j].dim, self.atom_to_basis_irreps[symbol_k].dim]})
#                 out.update({f'{symbol_j}_{symbol_k}':tpout})
#                 out1.update({f'{symbol_j}_{symbol_k}':out_js(symbol_j,symbol_k)})
#                 et.update({f'{symbol_j}_{symbol_k}':e3TensorDecomp(None,out_js_list=out1[f'{symbol_j}_{symbol_k}'])})
#         self.out=out
#         self.out1=out1
#         self.et=et
#         self.out_dim=out_dim
#
#         return
#
#     def deal(self,x,symbol_j,symbol_k):
#         o = []
#         num = 0
#         for mul, ir in self.tp_node.irreps_out.simplify():
#             for mul1, ir1 in self.out[f'{symbol_j}_{symbol_k}']:
#                 if ir==ir1:
#                     o.append(x[self.tp_node.irreps_out.slices()[num]][o3.Irreps(f'{mul1}x{ir1}').slices()[0]])
#
#             num += 1
#         return torch.cat(o, dim=-1)
#
#
#
#     def forward(self, node_features_batch: torch.Tensor, atom_data_batch_obj: Data,edge_vec,edge_length_gnn,node_attr_input: torch.Tensor) -> torch.Tensor:
#         if self.out is None:
#             self.reduce(node_features_batch.shape[0],atom_data_batch_obj.to_data_list()[0].atom_symbols_list)
#
#         edge_sh_attr = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(lmax_sh), edge_vec, True, normalization="component")
#
#         ptr = atom_data_batch_obj.ptr
#
#         for i in range(len(ptr) - 1):
#             node_features_sample = node_features_batch[ptr[i]:ptr[i + 1]]
#
#             num_atoms_sample = node_features_sample.shape[0]
#             sample_data = atom_data_batch_obj.to_data_list()[i]
#             atom_symbols_sample = sample_data.atom_symbols_list
#
#             extracted_block_features = []
#
#             for j in range(num_atoms_sample):
#                 for k in range(num_atoms_sample):  # 只计算上三角
#                     # 获取原子类型和对应的真实基组Irreps
#                     symbol_j = atom_symbols_sample[j]
#                     symbol_k = atom_symbols_sample[k]
#                     if j==k:
#                         block_features = self.tp_node(node_features_sample[j], node_features_sample[k])
#                         block_features = self.lin0(block_features, node_attr_input[j])
#                     else:
#                         fea_in = torch.cat([node_features_sample[j], node_features_sample[k]], dim=-1)
#                         fea_in = self.sort(fea_in)
#                         #block_features = self.tp_edge(node_features_sample[j], node_features_sample[k],self.fc(edge_length_gnn)[2*(i+j)-1].unsqueeze(0)).squeeze(0)
#                         block_features = self.tp_edge(fea_in, edge_sh_attr[2 * (i + j) - 1],self.fc(edge_length_gnn)[2*(i+j)-1].unsqueeze(0)).squeeze(0)
#                     block_features = self.deal(block_features,symbol_j,symbol_k)
#                     block_features = block_features.unsqueeze(0)
#
#                     block_features = self.et[f'{symbol_j}_{symbol_k}'].get_H(block_features)
#
#                     block_features = build_matrix_from_blocks(block_features,self.out_dim[f'{symbol_j}_{symbol_k}'])
#
#                     extracted_block_features.append(block_features)
#
#             final_sample_vector = build_matrix_from_blocks(extracted_block_features, [int(sum([i.shape[-1]*i.shape[-2] for i in extracted_block_features])**0.5)]*2)
#             #output_feature_vectors_batch.append(final_sample_vector)
#
#         return final_sample_vector-final_sample_vector.T
from torch_scatter import scatter_add


def coalesce_undirected_edges(edge_index: torch.Tensor,
                              edge_attr: torch.Tensor,
                              num_nodes: int = None) -> (torch.Tensor, torch.Tensor):
    """
    使用 torch_scatter 合并无向图中的重复边并累加其属性。
    """
    if num_nodes is None:
        num_nodes = int(torch.max(edge_index)) + 1

    # 规范化边
    sorted_edge_index, _ = torch.sort(edge_index, dim=0)
    row, col = sorted_edge_index

    # 创建唯一ID
    edge_uid = row * num_nodes + col

    # 使用 scatter_add 聚合
    unique_uid, inverse_indices = torch.unique(edge_uid, return_inverse=True)
    new_edge_attr = scatter_add(edge_attr, inverse_indices, dim=0, dim_size=unique_uid.size(0))

    # 还原边
    new_src = unique_uid // num_nodes
    new_dst = unique_uid % num_nodes
    new_edge_index = torch.stack([new_src, new_dst])

    return new_edge_index, new_edge_attr


class DensityMatrixFeatureBuilder(torch.nn.Module):
    """
    最终优化版：预先实例化所有可能的张量积模块，以提升性能并遵循最佳实践。
    """

    def __init__(self,
                 irreps_node_superset: o3.Irreps,num_radial_basis: int,
                 atom_to_basis_irreps: Dict[str, o3.Irreps],node_attr: o3.Irreps):
        super().__init__()
        self.node_attr=node_attr
        self.irreps_node_superset = o3.Irreps(irreps_node_superset)
        self.atom_to_basis_irreps = {k: o3.Irreps(v) for k, v in atom_to_basis_irreps.items()}
        #self.tp0=o3.FullTensorProduct(irreps_node_superset,irreps_node_superset)

        #self.tp_node=o3.FullyConnectedTensorProduct(irreps_node_superset,irreps_node_superset,self.tp0.irreps_out.simplify(),shared_weights=True)
        self.tp_node = o3.FullyConnectedTensorProduct(irreps_node_superset, irreps_node_superset,
                                                      irreps_node_superset, shared_weights=True)
        self.tp_edge=o3.FullyConnectedTensorProduct(irreps_node_superset,irreps_node_superset,irreps_node_superset,shared_weights=False)

        self.fc=e3nn_nn.FullyConnectedNet([num_radial_basis]  + [100] + [self.tp_edge.weight_numel if self.tp_edge else 0],act=torch.nn.functional.silu)
        self.lin0 = o3.FullyConnectedTensorProduct(irreps_node_superset, self.node_attr, irreps_node_superset)

        self.out=None

    def reduce(self,num,atom_symbols_sample):
        out=dict()
        out1=dict()
        et=dict()
        out_dim=dict()
        for j in range(num):
            for k in range(num):
                symbol_j = atom_symbols_sample[j]
                symbol_k = atom_symbols_sample[k]
                tpout=o3.FullTensorProduct(self.atom_to_basis_irreps[symbol_j],self.atom_to_basis_irreps[symbol_k]).irreps_out.simplify()
                out_dim.update({f'{symbol_j}_{symbol_k}': [self.atom_to_basis_irreps[symbol_j].dim, self.atom_to_basis_irreps[symbol_k].dim]})
                out.update({f'{symbol_j}_{symbol_k}':tpout})
                out1.update({f'{symbol_j}_{symbol_k}':out_js(symbol_j,symbol_k)})
                et.update({f'{symbol_j}_{symbol_k}':e3TensorDecomp(None,out_js_list=out1[f'{symbol_j}_{symbol_k}'])})
        self.out=out
        self.out1=out1
        self.et=et
        self.out_dim=out_dim

        return

    def deal(self,x,symbol_j,symbol_k):
        o = []
        num = 0
        for mul, ir in self.tp_node.irreps_out.simplify():
            for mul1, ir1 in self.out[f'{symbol_j}_{symbol_k}']:
                if ir==ir1:
                    o.append(x[self.tp_node.irreps_out.slices()[num]][o3.Irreps(f'{mul1}x{ir1}').slices()[0]])

            num += 1
        return torch.cat(o, dim=-1)



    def forward(self, node_features_batch: torch.Tensor,
                edge_features:torch.Tensor,edge_index: torch.Tensor,
                atom_data_batch_obj: Data,edge_length_gnn,node_attr_input: torch.Tensor) -> torch.Tensor:
        if self.out is None:
            self.reduce(node_features_batch.shape[0],atom_data_batch_obj.to_data_list()[0].atom_symbols_list)

        ptr = atom_data_batch_obj.ptr

        for i in range(len(ptr) - 1):
            node_features_sample = node_features_batch[ptr[i]:ptr[i + 1]]

            num_atoms_sample = node_features_sample.shape[0]
            sample_data = atom_data_batch_obj.to_data_list()[i]
            atom_symbols_sample = sample_data.atom_symbols_list

            extracted_block_features = []

            for j in range(num_atoms_sample):
                for k in range(num_atoms_sample):  # 只计算上三角
                    # 获取原子类型和对应的真实基组Irreps
                    symbol_j = atom_symbols_sample[j]
                    symbol_k = atom_symbols_sample[k]
                    if j==k:
                        block_features = self.tp_node(node_features_sample[j], node_features_sample[k])
                        block_features = self.lin0(block_features, node_attr_input[j])

                    else:
                        src_mask = (edge_index[0] == j)
                        dst_mask = (edge_index[1] == k)
                        final_mask = src_mask & dst_mask
                        # print(final_mask)
                        block_features = edge_features[final_mask,:].squeeze(0)
                        # print(block_features.shape)
                        # block_features = self.tp_edge(node_features_sample[j], node_features_sample[k],self.fc(edge_length_gnn)[2*(i+j)-1].unsqueeze(0)).squeeze(0)

                    block_features = self.deal(block_features,symbol_j,symbol_k)
                    block_features = block_features.unsqueeze(0)

                    block_features = self.et[f'{symbol_j}_{symbol_k}'].get_H(block_features)

                    block_features = build_matrix_from_blocks(block_features,self.out_dim[f'{symbol_j}_{symbol_k}'])

                    extracted_block_features.append(block_features)

            final_sample_vector = build_matrix_from_blocks(extracted_block_features, [int(sum([i.shape[-1]*i.shape[-2] for i in extracted_block_features])**0.5)]*2)
            #output_feature_vectors_batch.append(final_sample_vector)
            #print(final_sample_vector)

        return final_sample_vector


class AtomData(Data):  # Defined before
    def __init__(self, x=None, edge_index=None,edge_shift=None, pos=None, S_matrix=None, hcore_matrix=None,eri_matrix=None,
                 mol_str=None, atom_symbols_list=None, atom_basis_def_dict=None,
                 n_alpha=None, n_beta=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,edge_shift=edge_shift, pos=pos, **kwargs)
        self.S_matrix = S_matrix;
        self.hcore_matrix = hcore_matrix;
        self.eri_matrix = eri_matrix;
        self.mol_str = mol_str
        self.atom_symbols_list = atom_symbols_list;
        self.atom_basis_def_dict = atom_basis_def_dict
        self.n_alpha = n_alpha;
        self.n_beta = n_beta

    def __inc__(self, key, value, *args, **kwargs):
        return self.num_nodes if key == 'edge_index' else super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        return None if key in ['S_matrix', 'hcore_matrix','eri_matrix', 'mol_str', 'atom_symbols_list', 'atom_basis_def_dict',
                               'n_alpha', 'n_beta'] else super().__cat_dim__(key, value, *args, **kwargs)

import ase
from typing import List

class AtomsToGraphsConverter:
    def __init__(self, radial_cutoff: float = radius, max_atom_types_for_encoder: int = 4):
        self.radial_cutoff = radial_cutoff
        self.atom_encoder = None
        self.max_atom_types = max_atom_types_for_encoder
        self.fitted_dim = None
        self._fit_atom_encoder()

    def _fit_atom_encoder(self):
        self.fitted_dim = self.max_atom_types
        print(f"AtomsToGraphsConverter: Node feature 'x' will be {self.fitted_dim}-dim one-hot based on atomic number.")

    def _get_edges_from_positions(self,crystal,r_cut:int=5) -> torch.Tensor:
        # num_atoms = len(positions)
        # edges_src, edges_dst = [], []
        # for i in range(num_atoms):
        #     for j in range(i + 1, num_atoms):
        #         distance = np.linalg.norm(positions[i] - positions[j])
        #         if distance <= self.radial_cutoff and distance > 1e-5:
        #             edges_src.extend([i, j]);
        #             edges_dst.extend([j, i])
        edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=crystal, cutoff=r_cut,
                                                                        self_interaction=False)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0).to(device),
        edge_shift=torch.as_tensor(edge_shift, dtype=default_dtype).to(device),
        return edge_index, edge_shift

    def convert(self, atoms: Atoms, S_matrix: torch.Tensor, hcore_matrix: torch.Tensor,D:torch.Tensor,
                mol_str_repr: str, atom_symb_list: List[str], atom_basis_map: Dict,
                n_a: int, n_b: int) -> AtomData:
        positions = torch.tensor(atoms.get_positions(), dtype=default_dtype,device=device)
        lattice=(torch.as_tensor(atoms.cell.array).unsqueeze(0)).to(device)
        edge_index ,edge_shift= self._get_edges_from_positions(atoms,r_cut=r_cut)  # No .to(device) here, batching handles it

        atomic_nums = torch.tensor([atom.number for atom in atoms], dtype=torch.long,device=device)
        atomic_nums_0_indexed = atomic_nums - 1
        atomic_nums_clamped = torch.clamp(atomic_nums_0_indexed, 0, self.fitted_dim - 1)
        x = torch.nn.functional.one_hot(atomic_nums_clamped, num_classes=self.fitted_dim).to(default_dtype)
        x=x.to(device)

        return AtomData(x=x, pos=positions, edge_index=edge_index,edge_shift=edge_shift,lattice=lattice,
                        S_matrix=S_matrix, hcore_matrix=hcore_matrix,D=D,mol_str=mol_str_repr,
                        atom_symbols_list=atom_symb_list, atom_basis_def_dict=atom_basis_map,
                        n_alpha=torch.tensor(n_a, dtype=torch.long), n_beta=torch.tensor(n_b, dtype=torch.long),
                        # Store as tensors for batching
                        num_nodes=len(atoms)
                        )

from typing import Dict, Union, Tuple, List, Optional

from torch_geometric.data import Data, Dataset

class MoleculeDataset(Dataset):
    def __init__(self, molecule_data_list: List[Dict], radial_cutoff: float = radius, max_atom_types: int = 20):
        super().__init__()
        self.molecule_data_list = molecule_data_list
        self.converter = AtomsToGraphsConverter(radial_cutoff, max_atom_types)

    def len(self): return len(self.molecule_data_list)

    def get(self, idx):
        data_dict = self.molecule_data_list[idx]
        ase_atoms_obj = data_dict['ase_atoms']
        S_mat = data_dict['S'] if isinstance(data_dict['S'], torch.Tensor) else torch.tensor(data_dict['S'],
                                                                                             dtype=default_dtype)
        Hcore_mat = data_dict['Hcore'] if isinstance(data_dict['Hcore'], torch.Tensor) else torch.tensor(
            data_dict['Hcore'], dtype=default_dtype)
        # eri_mat=data_dict['eri'] if isinstance(data_dict['eri'], torch.Tensor) else torch.tensor(data_dict['eri'],
        #                                                                                      dtype=default_dtype)
        D=data_dict['D'] if isinstance(data_dict['D'], torch.Tensor) else torch.tensor(data_dict['D'],
                                                                                             dtype=default_dtype)
        return self.converter.convert(
            ase_atoms_obj, S_mat, Hcore_mat, D,data_dict['mol_str'],
            data_dict['atom_symbols_for_dm'], data_dict['atom_basis_def_for_dm'],
            data_dict['n_alpha'], data_dict['n_beta']
        )
class TorchNumInt_EnergyOnly:
    """
    一个TorchNumInt的版本，它只计算交换关联能量密度。
    这个版本经过修正，以匹配 libxc 中 VWN-5 泛函的计算结果。
    """

    def __init__(self, device_target=None):
        # 设定数据类型为双精度以匹配 libxc
        default_dtype = torch.float64

        self.device = device_target if device_target is not None else device
        Cx_unpol = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)

        self.Cx_spin_factor = torch.tensor(Cx_unpol * (2.0 ** (1.0 / 3.0)), dtype=default_dtype, device=self.device)

        # --- 顺磁性 (paramagnetic) 参数 (VWN-5, A_vwn[1] etc. in Maple) ---
        self.A_unpol = torch.tensor(0.0310907, dtype=default_dtype, device=self.device)
        self.x0_unpol = torch.tensor(-0.10498, dtype=default_dtype, device=self.device)
        self.b_unpol = torch.tensor(3.72744, dtype=default_dtype, device=self.device)
        self.c_unpol = torch.tensor(12.9352, dtype=default_dtype, device=self.device)

        # --- 铁磁性 (ferromagnetic) 参数 (VWN-5, A_vwn[2] etc. in Maple) ---
        self.A_ferro = torch.tensor(0.01554535, dtype=default_dtype, device=self.device)
        self.x0_ferro = torch.tensor(-0.32500, dtype=default_dtype, device=self.device)
        self.b_ferro = torch.tensor(7.06042, dtype=default_dtype, device=self.device)
        self.c_ferro = torch.tensor(18.0578, dtype=default_dtype, device=self.device)

        # =========================  主要修改点: 添加 RPA 参数 =========================
        # 根据您提供的 Maple 脚本，VWN 的插值还用到了第三套参数，
        # 这套参数对应于随机相位近似 (RPA) 的情况。
        # (A_vwn[3] etc. in Maple)
        self.A_rpa = torch.tensor(-1.0 / (6.0 * np.pi ** 2), dtype=default_dtype, device=self.device)
        self.x0_rpa = torch.tensor(-0.0047584, dtype=default_dtype, device=self.device)
        self.b_rpa = torch.tensor(1.13107, dtype=default_dtype, device=self.device)
        self.c_rpa = torch.tensor(13.0045, dtype=default_dtype, device=self.device)

        # fpp = 4/(9*(2^(1/3) - 1))
        self.fpp = torch.tensor(4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0)), dtype=default_dtype, device=self.device)
        # ===========================================================================

    def to(self, device_target):
        """将所有存储的张量移动到指定设备。"""
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device_target))
        self.device = device_target
        return self

    def _calculate_epsilon_c(self, rs: torch.Tensor, A, x0, b, c) -> torch.Tensor:
        """
        通用的 VWN 相关能量密度计算函数，对应 Maple 中的 f_aux。
        这个公式形式对顺磁性、铁磁性和RPA情况都适用，只是参数不同。
        """
        x = torch.sqrt(rs)
        X_val = rs + b * x + c

        # 使用 clamp 防止数值不稳定
        X_val_clamped = X_val#.clamp(min=1e-40)

        term1_log_inner = (rs / X_val_clamped)#.clamp(min=1e-40)
        term1 = torch.log(term1_log_inner)

        Q = torch.sqrt(4 * c - b * b)
        atan_denom = (2 * x + b)#.clamp(min=1e-40)
        term2 = 2 * b / Q * torch.atan(Q / atan_denom)

        X0_val = x0 ** 2 + b * x0 + c
        log_term_inner = ((x - x0).pow(2) / X_val_clamped)#.clamp(min=1e-40)
        term3_log = torch.log(log_term_inner)

        term3_atan_denom = (2 * x + b)#.clamp(min=1e-40)
        term3_atan = 2 * (b + 2 * x0) / Q * torch.atan(Q / term3_atan_denom)

        term3_factor = (b * x0) / X0_val

        epsilon_c = A * (term1 + term2 - term3_factor * (term3_log + term3_atan))
        return epsilon_c

    def eval_xc(self, rho_alpha: torch.Tensor, rho_beta: torch.Tensor) -> torch.Tensor:
        """
        简化版的eval_xc，只计算并返回交换关联能量密度。
        """
        dtype = self.Cx_spin_factor.dtype
        rho_alpha = rho_alpha.to(dtype)
        rho_beta = rho_beta.to(dtype)

        rho_alpha_clamped = rho_alpha#.clamp(min=1e-20)
        rho_beta_clamped = rho_beta#.clamp(min=1e-20)

        # 交换能部分
        ex_density_alpha = self.Cx_spin_factor * (rho_alpha_clamped ** (4.0 / 3.0))
        ex_density_beta = self.Cx_spin_factor * (rho_beta_clamped ** (4.0 / 3.0))
        ex_density_total = ex_density_alpha + ex_density_beta

        # 相关能部分
        rho_total = rho_alpha + rho_beta
        corr_mask = rho_total > 1e-20
        ec_density_total = torch.zeros_like(rho_total)

        if torch.any(corr_mask):
            rho_total_masked = rho_total[corr_mask]
            rs = (3.0 / (4.0 * np.pi * rho_total_masked)) ** (1.0 / 3.0)

            # 分别使用三套参数计算 epsilon_c
            ec_unpol = self._calculate_epsilon_c(rs, self.A_unpol, self.x0_unpol, self.b_unpol, self.c_unpol)
            ec_ferro = self._calculate_epsilon_c(rs, self.A_ferro, self.x0_ferro, self.b_ferro, self.c_ferro)
            ec_rpa = self._calculate_epsilon_c(rs, self.A_rpa, self.x0_rpa, self.b_rpa, self.c_rpa)

            zeta = (rho_alpha[corr_mask] - rho_beta[corr_mask]) / rho_total_masked

            # =========================  主要修改点: 修正自旋插值公式 =========================
            # 根据您提供的 Maple 脚本 `ker_lda_c_vwn` 更新插值公式。
            # 这与之前简单的线性插值不同。
            f_zeta = ((1 + zeta).pow(4.0 / 3.0) + (1 - zeta).pow(4.0 / 3.0) - 2.0) / (2.0 ** (4.0 / 3.0) - 2.0)

            dmc = ec_ferro - ec_unpol
            zeta_4 = zeta.pow(4)

            term_rpa = ec_rpa * f_zeta * (1 - zeta_4) / self.fpp
            term_dmc = dmc * f_zeta * zeta_4

            epsilon_c = ec_unpol + term_rpa + term_dmc
            # =================================================================================

            ec_density_total[corr_mask] = epsilon_c * rho_total_masked

        exc_density_total = ex_density_total + ec_density_total

        return exc_density_total

# class TorchNumInt:
#     """PyTorch implementation of PySCF's NumInt for XC calculations."""
#
#     def __init__(self, device=None):
#         # VWN parameters as tensors
#         self.A = torch.tensor(0.0310907, dtype=torch.float64, device=device)
#         self.x0 = torch.tensor(-0.10498, dtype=torch.float64, device=device)
#         self.b = torch.tensor(3.72744, dtype=torch.float64, device=device)
#         self.c = torch.tensor(12.9352, dtype=torch.float64, device=device)
#
#         # Exchange parameters
#         self.Cx = torch.tensor(-3 / 4 * (3 / np.pi) ** (1 / 3), dtype=torch.float64, device=device)
#
#     def eval_xc(self, xc_code: str, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Evaluate XC energy density and potential."""
#         if not rho.requires_grad:
#             rho = rho.detach().requires_grad_(True)
#
#         rho = rho.clamp(min=1e-32)
#
#         # Exchange part - LDA
#         ex = self.Cx * rho ** (4 / 3)
#
#         # VWN correlation part
#         rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
#         x = torch.sqrt(rs)
#
#         Q = torch.sqrt(4 * self.c - self.b * self.b)
#         X = rs + self.b * x + self.c
#         X0 = self.x0 * self.x0 + self.b * self.x0 + self.c
#
#         ec = self.A * (torch.log(rs / X) +
#                        2 * self.b / Q * torch.atan(Q / (2 * x + self.b)) -
#                        self.b * self.x0 / (X0 * (1 + self.x0 * self.x0 / X0)) *
#                        (torch.log((x - self.x0) * (x - self.x0) / X) +
#                         2 * (self.b + 2 * self.x0) / Q * torch.atan(Q / (2 * x + self.b))))
#
#         exc = ex + ec * rho
#         vxc = torch.autograd.grad(exc.sum(), rho, create_graph=True)[0]
#
#         return exc, vxc


from pyscf.pbc.dft import gen_grid, numint
import torch
import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf.pbc import df
from pyscf import ao2mo


def get_coulG_torch(cell: gto.Cell, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    使用PyTorch计算3D周期性体系的库仑核 (4π/|G|^2)。

    Args:
        cell: PySCF的Cell对象。
        device: Torch设备 ('cpu' or 'cuda')。
        dtype: Torch数据类型 (例如 torch.float64)。

    Returns:
        一个三维张量，包含了在FFT网格上每个G点的库仑核数值。
    """
    mesh = cell.mesh
    lattice_vectors = torch.from_numpy(cell.lattice_vectors()).to(device, dtype)

    # 1. 计算倒易矢量
    reciprocal_vectors = 2 * np.pi * torch.inverse(lattice_vectors).T

    # 2. 生成G矢量网格
    nx, ny, nz = mesh
    Gx = torch.fft.fftfreq(nx, d=1.0, device=device, dtype=dtype) * nx
    Gy = torch.fft.fftfreq(ny, d=1.0, device=device, dtype=dtype) * ny
    Gz = torch.fft.fftfreq(nz, d=1.0, device=device, dtype=dtype) * nz
    g_coords = torch.cartesian_prod(Gx, Gy, Gz)

    # 3. 将G矢量转换为笛卡尔坐标
    Gv_grid = torch.einsum('gi,ij->gj', g_coords, reciprocal_vectors)

    # 4. 计算 |G|^2
    absG2 = torch.einsum('gi,gi->g', Gv_grid, Gv_grid)

    # 5. 计算库仑核，并处理 G=0 奇点
    with torch.no_grad():
        coulG = 4 * np.pi / absG2
        # G=0 点的库仑核为0 (对于电中性体系)
        coulG[absG2 < 1e-9] = 0.0

    return coulG.reshape(nx, ny, nz)

from pyscf import ao2mo

# class DFTCalculator:
#     def __init__(self,cell: gto.Cell, kpts: np.ndarray, xc: str = xc):
#         """初始化DFT计算器"""
#
#         self.cell = cell
#
#         self.kpts = kpts
#         self.nkpts = len(kpts)
#
#         self.mf = dft.KUKS(self.cell,kpts=self.kpts)
#         self.mf.xc = xc
#
#         self.h_core = torch.from_numpy(mf.get_hcore()).to(device)
#         self.S = torch.from_numpy(mf.get_ovlp()).to(device)
#         eri = self.cell.intor('int2e')
#         eri = ao2mo.restore(1,eri,cell.nao)
#         self.eri=torch.from_numpy(eri).to(device)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
#         self.nuclear_repulsion = torch.tensor(cell.energy_nuc(), dtype=default_dtype).to(device)
#
#         # 设置数值积分
#         self.numint = TorchNumInt_EnergyOnly(device)
#         self.grid = gen_grid.UniformGrids(cell)
#         self.grid.build()
#         self.weights = torch.from_numpy(self.grid.weights).to(device)
#
#         #self.coulG = get_coulG_torch(self.cell, device, default_dtype)
#         # self.n_basis = self.S.shape[0]
#         # 获取AO值
#         ni = numint.KNumInt()
#         ao_kpts = ni.eval_ao(self.cell, self.grid.coords, kpts=self.kpts)
#         self.ao_values = torch.from_numpy(np.asarray(ao_kpts)).to(
#             self.device
#         )
#         print(f"Initialized KPointDFTCalculator with {self.nkpts} k-points on device: {device}")
#         print(f"Shape of ao_values tensor: {self.ao_values.shape}")
#     def to(self, device):
#         """将所有张量移动到指定设备"""
#
#         self.ao_values = self.ao_values.to(device)
#         self.weights = self.weights.to(device)
#         self.nuclear_repulsion = self.nuclear_repulsion.to(device)
#
#         # Move XC parameters
#         self.numint = TorchNumInt_EnergyOnly(device)
#         return self
#
#     def compute_density(self, P: torch.Tensor) -> torch.Tensor:
#         """计算实空间电子密度"""
#
#         rho = torch.zeros(self.grid.weights.shape[0], dtype=default_dtype, device=self.device)
#
#         for ik in range(self.nkpts):
#             # 公式: ρ_k(r) = Σ_μν P_μν(k) φ_μ*(r,k) φ_ν(r,k)
#             # 使用einsum高效计算
#             P_k = P[ik].to(self.S.dtype)
#
#             ao_k = self.ao_values[ik]
#             rho_k = torch.einsum('uv,gu,gv->g', P_k, ao_k.conj(), ao_k)
#             rho += rho_k.real
#
#         return rho/self.nkpts
#
#     def compute_hartree_energy(self, P: torch.Tensor) -> torch.Tensor:
#         """
#         根据实空间总电子密度，通过FFT计算哈特里能量。
#
#         Args:
#             rho: 实空间总电子密度，一个(ngrid,)形状的张量。
#
#         Returns:
#             哈特里能量 (一个标量)。
#         """
#
#         j_matrix=torch.einsum('ijkl,akl->aij', self.eri, P)
#         E_H=0.5*torch.einsum('kij,kji->',P,j_matrix).real/self.nkpts
#         return E_H
#
#     # def compute_j(self, P: torch.Tensor) -> torch.Tensor:
#     #     """计算库仑项"""
#     #     P = P.to(dtype=default_dtype)
#     #     return torch.einsum('ijkl,kl->ij', self.eri, P)
#
#     def compute_xc_energy(self, P:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """计算XC能量和势"""
#         #rho = self.compute_density(P)
#         rho = self.compute_density(P)
#         exc= self.numint.eval_xc(rho,rho)
#
#         # XC energy
#         e_xc = torch.dot(exc, self.weights)
#
#         # # XC potential matrix
#         # v_xc = torch.einsum('g,g,gp,gq->pq', vxc, self.weights, self.ao_values, self.ao_values)
#
#         return e_xc
#
#     def compute_energy(self, P_alpha:torch.Tensor) -> torch.Tensor:
#         """计算总能量"""
#         P_alpha = P_alpha
#
#         P_total = 2*P_alpha
#
#         # 一电子项
#         E_one = torch.einsum('kij,kji->', self.h_core, P_total).real / self.nkpts
#
#
#         E_H = self.compute_hartree_energy(P_total)
#         # Coulomb能量
#         #J = self.compute_j(P_total)
#         #E_H = 0.5 * torch.sum(P_total * J)
#
#         # XC能量
#         E_xc= self.compute_xc_energy(P_alpha)
#         print('E_one',E_one,'E_H',E_H,'E_xc',E_xc,'Enn',self.nuclear_repulsion)
#         return E_one + E_xc + E_H + self.nuclear_repulsion
import torch
import numpy as np
import h5py
from pyscf import lib
from pyscf.pbc import gto, dft, scf
from pyscf.pbc.dft import gen_grid, numint
from pyscf.pbc.df import df as pbc_df

import torch.nn as nn

class AntiSymmetricNet(nn.Module):
    """
    一个神经网络，接收一个向量，输出一个反对称矩阵。
    """

    def __init__(self, input_dim=128, matrix_size=32):
        super().__init__()
        self.matrix_size = matrix_size
        num_upper_tri_elements = (matrix_size * (matrix_size - 1)) // 2

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_upper_tri_elements)
        )

        triu_indices = torch.triu_indices(matrix_size, matrix_size, offset=1)
        self.register_buffer('triu_indices', triu_indices)

    def forward(self, x):
        batch_size = x.shape[0]
        upper_tri_elements = self.network(x)
        A = torch.zeros(batch_size, self.matrix_size, self.matrix_size, device=x.device)
        A[:, self.triu_indices[0], self.triu_indices[1]] = upper_tri_elements
        anti_symmetric_matrix = A - A.transpose(-2, -1)
        return anti_symmetric_matrix



class DFTCalculator:
    """
    A PyTorch-based DFT calculator for periodic systems that is end-to-end
    differentiable with respect to the density matrix.
    """

    def __init__(self, cell: gto.Cell, kpts: np.ndarray, xc: str = 'lda,vwn'):
        """Initializes the calculator by extracting necessary quantities from PySCF."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        default_dtype = torch.float64

        self.cell = cell
        self.kpts = kpts
        self.nkpts = len(kpts)

        # 1. Initialize PySCF objects to generate integrals
        mf = dft.KRKS(self.cell, kpts=self.kpts)
        mf.xc = xc
        # Crucially, initialize the GDF object
        mf.with_df = pbc_df.GDF(self.cell, kpts=self.kpts)
        # Run build() to generate the 3-center integral file
        mf.with_df.build()

        # 2. Extract Ewald-partitioned matrices and constants
        # Allow h_core and S to be complex by inferring dtype from numpy array
        hcore_kpts = np.asarray(mf.get_hcore())
        ovlp_kpts = np.asarray(mf.get_ovlp())
        self.h_core = torch.from_numpy(hcore_kpts).to(self.device)
        self.S = torch.from_numpy(ovlp_kpts).to(self.device)
        self.nuclear_repulsion = torch.tensor(cell.energy_nuc(), device=self.device, dtype=default_dtype)

        # 3. Load 3-center integrals (_cderi) from the HDF5 file
        self._cderi = self._load_cderi_from_file(mf.with_df)


        # 4. Set up the real-space grid for density and XC calculation
        self.grid = gen_grid.UniformGrids(self.cell)
        self.grid.build()
        self.weights = torch.from_numpy(self.grid.weights).to(self.device, default_dtype)

        ni = numint.KNumInt()
        ao_kpts = ni.eval_ao(self.cell, self.grid.coords, kpts=self.kpts)
        self.ao_values = torch.from_numpy(np.asarray(ao_kpts)).to(
            self.device, torch.cdouble if np.iscomplexobj(ao_kpts) else default_dtype
        )

        # 5. Initialize the differentiable XC functional
        self.numint_xc = TorchNumInt_EnergyOnly(self.device)

        print(f"Initialized KPointDFTCalculator with {self.nkpts} k-points on device: {self.device}")

    def _load_cderi_from_file(self, gdf_obj: pbc_df.GDF) -> list[torch.Tensor]:
        """
        Loads the 3-center integrals from the HDF5 file created by PySCF's GDF object.
        This function mimics the logic found in pyscf.pbc.df.df to handle
        segmented datasets and triangular storage.
        """
        cderi_tensors = []
        cderi_path = gdf_obj._cderi
        nao = self.cell.nao_nr()

        print(f"Loading 3-center integrals from: {cderi_path}")
        with h5py.File(cderi_path, 'r') as f:
            # Determine symmetry from the HDF5 file attributes
            if 'aosym' in f:
                aosym_val = f['aosym'][()]
                aosym = aosym_val.decode() if isinstance(aosym_val, bytes) else str(aosym_val)
            else:
                aosym = 's2'  # Default to s2 if not present
            dataname = gdf_obj._dataname

            for ik in range(self.nkpts):
                # For e_coul calculation, we only need the (k, k) blocks
                k_pair_label = f'{ik * self.nkpts + ik}'
                k_group_path = f'{dataname}/{k_pair_label}'

                if k_group_path not in f:
                    raise IOError(f"Dataset for k-point pair ({ik},{ik}) not found in {cderi_path}")

                k_group = f[k_group_path]

                # The data for a k-point pair can be split into multiple segments
                # We need to stack them horizontally.
                segments = [k_group[str(i)][:] for i in range(len(k_group))]
                cderi_k_packed = np.hstack(segments)

                naux = cderi_k_packed.shape[0]

                # For k=k, aosym is 's2' (lower triangular) to save space.
                # We must unpack it to a full square matrix.
                if aosym == 's2':
                    if np.iscomplexobj(cderi_k_packed):
                        # lib.unpack_tril does not support complex arrays directly
                        real_part = lib.unpack_tril(cderi_k_packed.real, axis=1)
                        imag_part = lib.unpack_tril(cderi_k_packed.imag, axis=1)
                        cderi_k_full = real_part + 1j * imag_part
                    else:
                        cderi_k_full = lib.unpack_tril(cderi_k_packed, axis=1)

                    # unpack_tril result has shape (naux, nao, nao), which is what we want
                else:  # aosym == 's1'
                    cderi_k_full = cderi_k_packed.reshape(naux, nao, nao)

                cderi_tensors.append(torch.from_numpy(cderi_k_full).to(self.device))

        # Ensure correct complex dtype for all tensors
        for i in range(len(cderi_tensors)):
            if cderi_tensors[i].dtype != torch.complex128:
                cderi_tensors[i] = cderi_tensors[i].to(torch.complex128)

        print("Finished loading 3-center integrals.")
        # if len(cderi_tensors)==1:
        #     return [cderi_tensors[0].real]
        # else:
        #     return cderi_tensors


        return cderi_tensors


    def compute_e_coul_from_dm(self, P: torch.Tensor) -> torch.Tensor:
        """
        Computes the Coulomb energy (e_coul) from the density matrix P.
        This implementation is differentiable and matches PySCF's GDF energy partition.
        """
        # Calculate the total auxiliary density coefficients by summing over all k-points.
        P=P.to(torch.complex128)
        rho_L_total = torch.zeros(self._cderi[0].shape[0], dtype=P.dtype, device=self.device)
        for ik in range(self.nkpts):
            P_k = P[ik]
            cderi_k = self._cderi[ik]  # Shape: (naux, nao, nao)
            # rho_L(k) = sum_uv P_vu(k) * (L|uv,k)
            # Note: PySCF's (L|uv) = <L|uv>. Its conjugate is not needed here.
            rho_L_k = torch.einsum('vu,Luv->L', P_k, cderi_k)
            rho_L_total += rho_L_k

        # The physical total density is the AVERAGE over k-points.
        # So the coefficients must be divided by nkpts before squaring.
        # if P.shape[0]==1:
        #     e_coul = 0.5 * (1.0 / self.nkpts ** 2) * torch.sum(rho_L_total.real ** 2 )
        # else:
        #     e_coul = 0.5 * (1.0 / self.nkpts ** 2) * torch.sum(rho_L_total.real ** 2 + rho_L_total.imag ** 2)
        e_coul = 0.5 * (1.0 / self.nkpts ** 2) * torch.sum(rho_L_total.real ** 2 + rho_L_total.imag ** 2)

        return e_coul

    def compute_density(self, P: torch.Tensor) -> torch.Tensor:
        """Computes the real-space electron density from the density matrix P."""
        rho = torch.zeros(self.grid.weights.shape[0], dtype=torch.float64, device=self.device)
        for ik in range(self.nkpts):
            P_k = P[ik]
            ao_k = self.ao_values[ik]
            rho_k = torch.einsum('uv,gu,gv->g', P_k, ao_k.conj(), ao_k).real
            rho += rho_k
        return rho / self.nkpts

    def compute_xc_energy(self, rho: torch.Tensor) -> torch.Tensor:
        """Computes the XC energy from the real-space density rho."""
        exc_density = self.numint_xc.eval_xc(rho / 2, rho / 2)
        # E_xc = integral(exc_density) dr
        e_xc = torch.dot(exc_density, self.weights)
        return e_xc

    def compute_energy(self, P_alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes the total DFT energy for a restricted calculation.
        The energy components are defined consistently with PySCF's GDF framework.
        """
        # Assumes restricted calculation: P_total = 2 * P_alpha
        P_total = 2 * P_alpha

        # if P_total.shape[0]==1:
        #     pass
        # else:
        #     P_total = torch.tensor(P_total,dtype=torch.complex128).unsqueeze(0)

        # 1. One-electron energy (from Ewald-partitioned h_core)
        E_one = torch.einsum('kij,kji->', self.h_core, P_total).real / self.nkpts

        # 2. Coulomb energy (from our differentiable e_coul implementation)
        E_coul = self.compute_e_coul_from_dm(P_total)

        # 3. Exchange-correlation energy
        rho = self.compute_density(P_total)
        E_xc = self.compute_xc_energy(rho)

        # 4. Nuclear repulsion energy (from Ewald summation)
        E_nuc = self.nuclear_repulsion

        total_energy = E_one + E_coul + E_xc + E_nuc

        # print(
        #     f"E_one: {E_one.item():.8f}, E_coul: {E_coul.item():.8f}, E_xc: {E_xc.item():.8f}, E_nuc: {E_nuc.item():.8f}")
        # print(f"E_total: {total_energy.item():.8f}")

        return {'total_energy': total_energy, 'one_electron': E_one, 'coulomb': E_coul, 'xc': E_xc, 'nuclear_repulsion': E_nuc}


def train(model: DensityPredictingGNN,num_epochs: int, dataloader: PyGDataLoader,primal_optimizer: torch.optim.Optimizer,
          scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],device: torch.device):

    data_sample = batch.to_data_list()[0]
    S = data_sample.S_matrix
    n_basis = SS.shape[1]

    dft_calculator = DFTCalculator(data_sample['mol_str'], kpts=kpts)

    for epoch in range(num_epochs):
        model.train()
        for pyg_batch in dataloader:

            X_full = model(pyg_batch)
            X_full = X_full-X_full.transpose(-1, -2)

            # X_full = ppx(X_full)
            #
            # X_full = tr.real2k(X_full)


            # X = rebundant_matrix(X)
            num_graphs = pyg_batch.num_graphs
            total_loss = 0.0
            primal_optimizer.zero_grad()
            for i in range(num_graphs):

                # data_sample = pyg_batch.to_data_list()[i]
                #
                # S = data_sample.S_matrix
                #
                # n_basis = S.shape[0]
                #
                #
                # # 计算DFT能量 (你的目标函数)
                # dft_calculator = DFTCalculator(data_sample['mol_str'],kpts=kpts)  # 重新实例化或从data中获取
                # dft_calculator.S = data_sample['S_matrix']
                # dft_calculator.h_core = data_sample['hcore_matrix']
                # dft_calculator.eri = data_sample['eri_matrix'].squeeze(0)

                # P_op = torch.einsum('ijk,ikl->ijl', P0, dft_calculator.S)
                P_op = torch.einsum('ij,jk->ik', P0, SS)
                Q_op = torch.eye(n_basis, dtype=default_dtype, device=device) - P_op

                # X_proj = P_op @ X_full @ Q_op.T + Q_op @ X_full @ P_op.T
                #
                # exp_neg_XS = torch.linalg.matrix_exp(-X_proj @ dft_calculator.S)
                # exp_SX = torch.linalg.matrix_exp(dft_calculator.S @ X_proj)
                # P_new = exp_neg_XS @ P0 @ exp_SX

                # X_proj = (torch.einsum('bik,bkj,bjl->bil', P_op, X_full, Q_op) +
                #           torch.einsum('bik,bkj,bjl->bil', Q_op, X_full, P_op))

                X_proj = (torch.einsum('ik,kj,lj->il', P_op, X_full, Q_op) +
                          torch.einsum('ik,kj,lj->il', Q_op, X_full, P_op))

                # exp_neg_XS = torch.linalg.matrix_exp(-X_proj @ dft_calculator.S)
                exp_neg_XS = torch.linalg.matrix_exp(-torch.einsum('ik,kj->ij', X_proj, SS))

                # exp_SX = torch.linalg.matrix_exp(dft_calculator.S @ X_proj)
                exp_SX = torch.linalg.matrix_exp(torch.einsum('ik,kj->ij', SS, X_proj))

                # P_new = exp_neg_XS @ P0 @ exp_SX
                P_new = torch.einsum('ik,kl,lj->ij', exp_neg_XS, P0, exp_SX)

                P_new = ppD(P_new*nkpts)
                P_new = tr.real2k(P_new)

                dft_energy = dft_calculator.compute_energy(P_new)['total_energy']
                total_loss += dft_energy

            total_loss.backward()
            primal_optimizer.step()
            if scheduler:
                scheduler.step(total_loss)
            out_energy.append(total_loss.item())
            out_d.append(P_new)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}, Energy (Loss): {total_loss.item():.8f}")

    print("\n--- Optimization Finished ---")
    # Compare with PySCF's converged energy
    # mf_pyscf = dft.KUKS(dft_calculator.cell,kpts=kpts)
    # mf_pyscf.xc = 'lda,vwn'
    # pyscf_energy = mf_pyscf.kernel()
    print(f"Final NN-Optimized Energy: {min(out_energy):.8f}")
    print(f"PySCF Converged Energy:    {mf.e_tot:.8f}")

def calculate_D(X_full: torch.Tensor,P0:torch.Tensor):


    X_full = X_full - X_full.transpose(-1, -2)
    n_basis = X_full.shape[-1]
    P_op = torch.einsum('ij,jk->ik', P0, SS)
    Q_op = torch.eye(n_basis, dtype=default_dtype, device=device) - P_op

    # X_proj = P_op @ X_full @ Q_op.T + Q_op @ X_full @ P_op.T
    #
    # exp_neg_XS = torch.linalg.matrix_exp(-X_proj @ dft_calculator.S)
    # exp_SX = torch.linalg.matrix_exp(dft_calculator.S @ X_proj)
    # P_new = exp_neg_XS @ P0 @ exp_SX

    # X_proj = (torch.einsum('bik,bkj,bjl->bil', P_op, X_full, Q_op) +
    #           torch.einsum('bik,bkj,bjl->bil', Q_op, X_full, P_op))

    X_proj = (torch.einsum('ik,kj,lj->il', P_op, X_full, Q_op) +
              torch.einsum('ik,kj,lj->il', Q_op, X_full, P_op))

    # exp_neg_XS = torch.linalg.matrix_exp(-X_proj @ dft_calculator.S)
    exp_neg_XS = torch.linalg.matrix_exp(-torch.einsum('ik,kj->ij', X_proj, SS))

    # exp_SX = torch.linalg.matrix_exp(dft_calculator.S @ X_proj)
    exp_SX = torch.linalg.matrix_exp(torch.einsum('ik,kj->ij', SS, X_proj))

    # P_new = exp_neg_XS @ P0 @ exp_SX
    P_new = torch.einsum('ik,kl,lj->ij', exp_neg_XS, P0, exp_SX)

    P_new = ppD(P_new * nkpts)
    P_new = tr.real2k(P_new)
    # P_new = (P_new+P_new.transpose(-1,-2))/2
    return P_new

def calculate_energy(X_full: torch.Tensor):
    P_new = calculate_D(X_full,P0)

    dft_energy = dft_calculator.compute_energy(P_new)
    return dft_energy

def trainD(modelD: DensityPredictingGNN,num_epochs: int,dataloader: PyGDataLoader,primal_optimizer: torch.optim.Optimizer,
           scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],device: torch.device):
    data_sample = batch.to_data_list()[0]
    S = data_sample.S_matrix


    for epoch in range(num_epochs):
        modelD.train()
        for pyg_batch in dataloader:

            D = modelD(pyg_batch)
            # D = (D + D.transpose(-1, -2)) / 2

            D = trace(D).squeeze(0)
            D = (D + D.T) / 2
            # D = ppD(D)
            #
            #
            #
            # D = tr.real2k(D)
            # D = (D + D.T)/2
            # D = trace(D)


            num_graphs = pyg_batch.num_graphs
            total_loss = 0.0
            primal_optimizer.zero_grad()
            for i in range(num_graphs):
                D_squared =D@SS@D
                # D_squared = torch.einsum('ik,kj,jl->il',D,SS,D)
                mse= torch.nn.functional.mse_loss(D_squared,D)
                # msei = torch.nn.functional.mse_loss(D_squared.imag,D.imag)
                # mse = mser+msei
                total_loss+= mse

            total_loss.backward()
            primal_optimizer.step()
            if scheduler:
                scheduler.step(total_loss)
            out.append(total_loss.item())

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}, Idop (Loss): {total_loss.item():.8f}")
        if torch.allclose(D_squared,D,atol=1e-10):
            break
    print("\n--- Optimization Finished ---")
    # Compare with PySCF's converged energy
    # mf_pyscf = dft.KUKS(dft_calculator.cell,kpts=kpts)
    # mf_pyscf.xc = 'lda,vwn'
    # pyscf_energy = mf_pyscf.kernel()
    print(f"Final NN-Optimized Energy: {min(out):.8f}")
    #print(f"PySCF Converged Energy:    {mf.e_tot:.8f}")



from torch_geometric.loader import DataLoader as PyGDataLoader
import scipy
import pandas as pd
from pandarallel import pandarallel

if __name__ == '__main__':
    print("Starting GNN for DFT Density Matrix Prediction...")

    np.set_printoptions(precision=3, suppress=True)

    aa = 2.46
    base_structure = [
        # 第一个单胞的2个碳原子
        ['C', [0.0, 0.0, 0.0]],
        ['C', [aa/2, aa/(2*3**0.5), 0.0]],

        # 第二个单胞的2个碳原子 (y方向平移)
        ['C', [aa/2, aa*3**0.5/2, 0.0]],
        ['C', [aa, aa*3**0.5/2+aa/(2*3**0.5), 0.0]],

        # 第三个单胞的2个碳原子 (x方向平移)
        ['C', [aa, 0.0, 0.0]],
        ['C', [aa + aa / 2, aa / (2 * 3 ** 0.5), 0.0]],
        # 第四个单胞的2个碳原子 (xy方向同时平移)
        ['C', [aa+aa/2, aa*3**0.5/2, 0.0]],
        ['C', [2*aa, aa*3**0.5/2+aa/(2*3**0.5), 0.0]]
    ]
    a = [[2*aa, 0.0, 0.0],  # 2 * 2.46
              [aa, aa*3**0.5, 0.0],  # 2 * 2.13042249330972
              [0.0, 0.0, 12.0]]

    cells = gto.Cell()
    cells.a = a
    cells.atom = base_structure
    cells.basis = basis
    cells.pseudo = pseudo
    cells.unit = 'A'
    cells.verbose = -1
    cells.build(symmetry=False)

    kpts = cells.make_kpts([1,1,1])
    mfs = dft.KUKS(cells,kpts=kpts)
    # cell.precision = 1e-10

    symbols = []
    # 创建一个空的原子坐标列表
    positions = []

    for atom_info in base_structure:
        symbols.append(atom_info[0])
        positions.append(atom_info[1])

    graphene_supercell = Atoms(symbols=symbols,
                               positions=positions,
                               cell=a,
                               pbc=True)


    aa = 2.46
    cell = gto.Cell()

    # 晶格向量
    a = [
        [aa, 0.0, 0.0],
        [aa/ 2, aa * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, 12.0]  # Z方向的真空层保持不变
    ]

    # 原胞中的两个碳原子的笛卡尔坐标
    base_structure  = [
        ['C', [0.0, 0.0, 0.0]],
        ['C', [aa / 2, aa / (2 * np.sqrt(3)), 0.0]]
    ]

    # cell.dimension = 2
    # cell.mesh = [40, 40, 150]
    # 单胞晶格矢量
    cell.a = a
    cell.atom = base_structure
    cell.basis =basis
    cell.pseudo = pseudo
    cell.max_memory = 8000
    cell.unit = 'A'
    # cell.precision = 1e-5
    cell.verbose = 4



    cell.build(symmetry=False)
    # cell.build()

    k_mesh=[2, 2, 1]
    nkx,nky,nkz = k_mesh
    kpts=cell.make_kpts(k_mesh)
    nkpts=nkx*nky*nkz
    #kpts=np.array([[0.676, 0.39 , 0.   ]])


    mf=dft.KUKS(cell,kpts=kpts)
    mf.with_df = pbc_df.GDF(cell, kpts=kpts)

    mf.xc = 'lda,vwn'
    # mf.kernel()

    tr = trans(cell=cell,k_mesh=k_mesh,dimension=2,)
    ppD = post_processing_D(k_mesh)

    ppx = post_processing_X(k_mesh)
    # mf.max_cycle=1
    # mf.kernel()
    # P0=torch.from_numpy(mf.make_rdm1()[0]).to(device)
    # mf.kernel()
    #mf.build(symmetry=False)
    # 创建一个空的原子符号列表
    # symbols = []
    # # 创建一个空的原子坐标列表
    # positions = []
    #
    # for atom_info in base_structure:
    #     symbols.append(atom_info[0])
    #     positions.append(atom_info[1])
    #
    # graphene_supercell = Atoms(symbols=symbols,
    #                            positions=positions,
    #                            cell=a,
    #                            pbc=True)



    ase_=graphene_supercell

    s_matrix = mf.get_ovlp()
    # s_matrix = s_matrix+s_matrix.transpose(0,2,1)
    # s_matrix = s_matrix/2

    S=torch.tensor(s_matrix,device=device)
    hcore = mf.get_hcore()
    Hcore = torch.tensor(hcore,device=device)
    # eri = df.FFTDF(cell).get_eri()
    # eri = ao2mo.restore(1, eri, cell.nao)
    # eri = torch.tensor(eri, device=device)

    n_alpha, n_beta = mf.nelec
    atom_symbols_dm = ['C']*len(base_structure)*nkx*nky*nkz
    atom_basis_def_gp_dm = {'C':[o3.Irreps("0e"), o3.Irreps("1o")]}


    #P0=torch.as_tensor(mf.get_init_guess(key='1e')[0]).to(device)
    #P0 = torch.from_numpy(mf.get_init_guess(key='1e')[0]).to(device)
    # j_matrix = mf.get_j(dm=P0)
    # hcore = mf.get_hcore()
    # k_matrix = mf.get_k(dm=P0)
    # fock_with_hartree = hcore + j_matrix+k_matrix
    #
    #
    # e,C_np=scipy.linalg.eigh(hcore[0],s_matrix[0])
    # n_occ = n_alpha
    # #C_occ_np = C_np[:, :n_occ]
    # #C_occ_np = C_np[:, [0,1,2,-1]]
    # C_occ_np = C_np[:, [0, 1, 2, 3,4,5,6,7,8,9,10,11,-1,-5,-6,-7]]
    # C_occ = torch.tensor(C_occ_np, device=device)
    # P0 = C_occ @ C_occ.T
    # P0=P0.unsqueeze(0)



    # mol_str_h2 = "H 0 0 0; H 0 0 0.74"
    # ase_h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
    # basis_h2 = basis
    # pyscf_mol_h2 = gto.M(atom=mol_str_h2, basis=basis_h2, verbose=0)
    # pyscf_mf_h2=scf.UKS(pyscf_mol_h2)
    # S = torch.tensor(pyscf_mf_h2.get_ovlp(), dtype=default_dtype,device=device)
    # Hcore_h2 = torch.tensor(pyscf_mf_h2.get_hcore(), dtype=default_dtype,device=device)
    # eri_h2 = torch.tensor(pyscf_mol_h2.intor('int2e'), dtype=default_dtype, device=device)
    # n_alpha, n_beta = pyscf_mf_h2.nelec
    # atom_symbols_dm = ['H', 'H']
    # atom_basis_def_h2_dm = {'H': [o3.Irreps("0e")]}
    #
    # e,C_np=scipy.linalg.eigh(pyscf_mf_h2.get_hcore(),pyscf_mf_h2.get_ovlp())
    # n_occ = n_alpha
    # C_occ_np = C_np[:, :n_occ]
    # C_occ = torch.tensor(C_occ_np, dtype=default_dtype, device=device)
    # # Construct initial idempotent single-spin density matrix: P₀ = C_occ @ C_occ.T
    # P0 = C_occ @ C_occ.T
    # pyscf_mf_h2.kernel()
    # P0=torch.tensor(pyscf_mf_h2.make_rdm1()[0],dtype=default_dtype,device=device)

    raw_data_listD = [
        # {'mol_str': mol_str_h2o, 'basis': basis_h2o, 'S': S, 'Hcore': Hcore_h2o, 'D':P0,
        #  'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_h2o,
        #  'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_h2o_dm,'eri': eri_h2o},
        {'mol_str': cell, 'basis': basis, 'S': S, 'Hcore': Hcore, 'D': S,'lattice':a,
         'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_,
         'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_gp_dm},
       # {'mol_str': mol_str_h2, 'basis': basis_h2, 'S': S, 'Hcore': Hcore_h2,'D':P0,
       #  'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_h2,
       #  'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_h2_dm,'eri': eri_h2},
    ]


    out=[]
    atom_to_basis_map_str = dict()
    for atom in atom_symbols_dm:
        atom_to_basis_map_str.update(net_output(atom))

    irreps_core_gnn_node_out_str=atom_to_basis_map_str[max(atom_to_basis_map_str,key=lambda k:atom_to_basis_map_str[k].dim)]
    datasetD = MoleculeDataset(raw_data_listD, radial_cutoff=radius, max_atom_types=10)
    dataloaderD = PyGDataLoader(datasetD, batch_size=batch_size, shuffle=False)  # Reduced batch size for testing

    for batch in dataloaderD:
        print(batch)
        break

    node_out=o3.FullTensorProduct(irreps_core_gnn_node_out_str,irreps_core_gnn_node_out_str)

    node_attr_dim = datasetD.converter.fitted_dim



    modelD = DensityPredictingGNN(
        irreps_node_attr_in_str=f"{node_attr_dim}x0e", irreps_node_embed_in_str="10x0e",
        # Reduced embed for speed
        irreps_query_str="2x0e+1o", irreps_key_str="2x0e+1o",
        irreps_node_hidden_str="8x0e+4x1o+2x2e+1x3o+1x4e", irreps_core_gnn_node_out_str=node_out.irreps_out.simplify()+o3.Irreps('3o'),#irreps_core_gnn_node_out_str+o3.Irreps('2e'),
        num_attn_layers=layers, num_radial_basis=10,
        atom_to_basis_map=atom_to_basis_map_str
    ).to(device)

    # trace = BatchedConstrainedTraceLayer(S, n_alpha)
    SS = mfs.get_ovlp()[0]
    SS = torch.from_numpy(SS).to(device)
    trace = ConstrainedTraceLayer(SS, mfs.nelec[0])

    primal_optimizerD = torch.optim.Adam(modelD.parameters(), lr=3e-2)

    schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(primal_optimizerD, mode='min', factor=0.995, patience=5,
                                                          verbose=True)

    trainD(num_epochs=1000, modelD=modelD, dataloader=dataloaderD, primal_optimizer=primal_optimizerD,scheduler=schedulerD, device=device)


    # P0 = modelD(batch)+modelD(batch).T
    # P0 = P0/2
    P0 = modelD(batch).detach()
    P0 = (P0+P0.transpose(-2,-1))/2
    P0 = trace(P0).squeeze(0)
    # P0 = ppD(P0)
    # P0 = tr.real2k(P0)
    # P0 = trace(P0).detach()+trace(P0).detach().transpose(-2,-1)
    # P0 = P0/2
    #P0 = P0.unsqueeze(0)

    # dft_calculator.S = data_sample['S_matrix']
    # dft_calculator.h_core = data_sample['hcore_matrix']
    # dft_calculator.eri = data_sample['eri_matrix'].squeeze(0)




    raw_data_list = [
        # {'mol_str': mol_str_h2o, 'basis': basis_h2o, 'S': S, 'Hcore': Hcore_h2o, 'D':P0,
        #  'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_h2o,
        #  'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_h2o_dm,'eri': eri_h2o},
        {'mol_str': cell, 'basis': basis, 'S': S, 'Hcore': Hcore, 'D': P0,'lattice':a,
         'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_,
         'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_gp_dm},
       # {'mol_str': mol_str_h2, 'basis': basis_h2, 'S': S, 'Hcore': Hcore_h2,'D':P0,
       #  'n_alpha': n_alpha, 'n_beta': n_beta, 'ase_atoms': ase_h2,
       #  'atom_symbols_for_dm': atom_symbols_dm, 'atom_basis_def_for_dm': atom_basis_def_h2_dm,'eri': eri_h2},
    ]

    dataset = MoleculeDataset(raw_data_list, radial_cutoff=radius, max_atom_types=10)
    dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)  # Reduced batch size for testing

    node_out=o3.FullTensorProduct(irreps_core_gnn_node_out_str,irreps_core_gnn_node_out_str)

    out_energy=[]
    out_d=[]

    model = DensityPredictingGNN(
        irreps_node_attr_in_str=f"{node_attr_dim}x0e", irreps_node_embed_in_str="10x0e",
        # Reduced embed for speed
        irreps_query_str="2x0e+1x1o", irreps_key_str="2x0e+1x1o",
        irreps_node_hidden_str="8x0e+8x0o+4x1o+4x1e+2x2e+2x2o+1x3o+1x4e", irreps_core_gnn_node_out_str=node_out.irreps_out.simplify()+o3.Irreps('3o'),#irreps_core_gnn_node_out_str+o3.Irreps('2e'),
        num_attn_layers=layers, num_radial_basis=10,
        atom_to_basis_map=atom_to_basis_map_str
    ).to(device)

    for batch in dataloader:
        print(batch)
        break

    data_sample = batch.to_data_list()[0]
    dft_calculator = DFTCalculator(data_sample['mol_str'], kpts=kpts) # 重新实例化或从data中获取

    primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(primal_optimizer, mode='min', factor=0.995, patience=5,
                                                          verbose=True)
    num_epochs = 200
    train(num_epochs=num_epochs, model=model, dataloader=dataloader, primal_optimizer=primal_optimizer,scheduler=scheduler, device=device)
    # Start Cooper ALM training
    # admm_train_loop(
    #     num_epochs=1000,  # Was num_outer_alm_epochs
    #     model=model,
    #     dataloader=dataloader,
    #     primal_optimizer=primal_optimizer,
    #     rho=5,
    #     device=device
    # )

    print("\nIllustrative Energy training finished.")

    mf.kernel()
    data=pd.DataFrame()
    pandarallel.initialize(progress_bar=True)
    epoch=list(range(num_epochs))
    data['epoch']=epoch
    out_diff_energy=[i-mf.e_tot for i in out_energy]
    # data['out_energy']=out_energy
    data['out_diff_energy']=out_diff_energy
    dmf=mf.make_rdm1()[0]

    out_diff_d=[2*np.abs((i.detach().cpu().numpy()-dmf)).sum()/(nkpts*(4*len(cell.atom))**2) for i in out_d]

    data['out_diff_dm']=out_diff_d

    rho=mf.get_rho()

    data['dm']=[i.detach().cpu().numpy() for i in out_d]

    rho_pred=[]

    for i in tqdm(range(num_epochs)):
        rho_pred.append(mf.get_rho(2*data['dm'][i]))


    rho_diff = [i-rho for i in rho_pred]
    rho_diff = [np.abs(i).sum()*cell.vol/rho.shape[0] for i in rho_diff]

    data['dm'] = rho_diff
    # data['rho']=data['dm'].parallel_apply(lambda x:mf.get_rho(2*x))