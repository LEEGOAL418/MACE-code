import os
import time
import argparse
from ase.io import read, write
from ase.optimize import LBFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.io.trajectory import Trajectory
from mace.calculators.mace import MACECalculator
from ase.constraints import FixedLine


def optimize_structure(
    model_path,
    init_structure_path,
    output_dir,  # 输出文件的根目录
    optimizer="LBFGS",  # 默认使用 LBFGS
    fmax=0.01,
    constrain_c_atoms=False  # 布尔值参数，控制是否施加约束
):
    """
    优化原子结构并记录优化过程

    :param model_path: MACE 模型路径
    :param init_structure_path: 初始结构文件路径
    :param output_dir: 输出文件的根目录
    :param optimizer: 优化算法选择 ("LBFGS" 或 "CG")
    :param fmax: 最大力收敛标准 (eV/Å)
    :param constrain_c_atoms: 布尔值参数，控制是否对 C 原子施加约束（只允许在 z 方向移动）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取初始结构的基础文件名
    base_name = os.path.splitext(os.path.basename(init_structure_path))[0]

    # 动态生成输出文件路径
    traj_file_path = os.path.join(output_dir, "trajectory", f"{base_name}_{optimizer}_traj.traj")
    relaxed_structure_path = os.path.join(output_dir, "optimizedcell", f"{base_name}_{optimizer}_relaxed.extxyz")
    steps_output_path = os.path.join(output_dir, "steps", f"{base_name}_{optimizer}_steps.extxyz")

    # 确保各子目录存在
    os.makedirs(os.path.dirname(traj_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(relaxed_structure_path), exist_ok=True)
    os.makedirs(os.path.dirname(steps_output_path), exist_ok=True)

    # 加载 MACE 模型
    calculator = MACECalculator(model_path=model_path, device='cuda')

    # 读取初始结构文件
    init_conf = read(init_structure_path)

    # 为初始结构设置 MACE 计算器
    init_conf.set_calculator(calculator)

    # 施加约束（如果指定）
    if constrain_c_atoms:
        # 限制所有碳原子仅在z方向上移动
        carbon_indices = [atom.index for atom in init_conf if atom.symbol == 'C']
        if carbon_indices:
            constraints = FixedLine(indices=carbon_indices, direction=[0, 0, 1])
            init_conf.set_constraint(constraints)
            print("Applied constraints on carbon atoms to move only along z direction.")
        else:
            print("No carbon atoms found to apply constraints.")

    # 打印初始能量
    initial_energy = init_conf.get_potential_energy()
    print(f"Initial Energy: {initial_energy:.6f} eV")

    # 记录优化开始时间
    start_time = time.time()

    # 创建 .traj 文件记录优化过程
    traj = Trajectory(traj_file_path, 'w', init_conf)

    # 选择优化器
    if optimizer == "LBFGS":
        dyn = LBFGS(init_conf, trajectory=traj)
        print("Using LBFGS optimizer.")
    elif optimizer == "CG":
        dyn = SciPyFminCG(init_conf, trajectory=traj)
        print("Using SciPy CG optimizer.")
    else:
        raise ValueError("Invalid optimizer selected. Choose 'LBFGS' or 'CG'.")

    # 运行优化
    dyn.run(fmax=fmax)  # 优化停止条件

    # 记录优化结束时间
    end_time = time.time()

    # 打印优化总时间
    total_time = end_time - start_time
    print(f"Optimization completed in {total_time:.2f} seconds.")

    # 打印优化后的能量
    optimized_energy = init_conf.get_potential_energy()
    print(f"Optimized Energy: {optimized_energy:.6f} eV")

    # 输出优化后的结构到文件
    write(relaxed_structure_path, init_conf)  # 保存优化后的结构
    print(f"Relaxed structure saved to: {relaxed_structure_path}")

    # 将轨迹文件中的所有优化步骤保存到一个 .extxyz 文件
    steps = read(traj_file_path, index=':')  # 读取轨迹文件中的所有步骤
    with open(steps_output_path, 'w') as f:
        for step, atoms in enumerate(steps):
            write(f, atoms, format='extxyz')  # 写入每一步的信息
    print(f"Optimization steps saved to: {steps_output_path}")

    print("Optimization finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize atomic structures using MACE and ASE.")
    parser.add_argument("--model_path", required=True, help="Path to the MACE model file.")
    parser.add_argument("--init_structure_path", required=True, help="Path to the initial structure file.")
    parser.add_argument("--output_dir", required=True, help="Root directory for output files.")
    parser.add_argument(
        "--optimizer",
        choices=["LBFGS", "CG"],
        default="LBFGS",
        help="Optimization algorithm to use (default: LBFGS)."
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.01,
        help="Maximum force convergence criteria (default: 0.01 eV/Å)."
    )
    parser.add_argument(
        "--constrain_c_atoms",
        action='store_true',
        help="Apply constraints on C atoms to move only along z direction."
    )

    args = parser.parse_args()

    optimize_structure(
        model_path=args.model_path,
        init_structure_path=args.init_structure_path,
        output_dir=args.output_dir,
        optimizer=args.optimizer,
        fmax=args.fmax,
        constrain_c_atoms=args.constrain_c_atoms
    )
