#!/usr/bin/env python3
# error_calculate.py

import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict

from ase.io import read
import numpy as np
import pandas as pd

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        # logging.FileHandler("error_calculate.log")  # 可选：输出到文件
    ]
)


def read_true_data(true_file: str) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """
    读取真实数据集，返回能量、力和原子数量。

    参数:
        true_file (str): 真实数据集文件路径

    返回:
        Tuple[np.ndarray, List[np.ndarray], List[int]]: 能量数组，力数组列表，原子数量列表
    """
    try:
        structures = read(true_file, index=':')
        energies = []
        forces = []
        atom_counts = []
        for i, atoms in enumerate(structures):
            energy = atoms.get_potential_energy()
            if not isinstance(energy, (float, int)):
                logging.error(f"Structure {i} in {true_file} has non-scalar energy: {energy}")
                sys.exit(1)
            energies.append(float(energy))

            force = atoms.get_forces()
            if not isinstance(force, np.ndarray):
                logging.error(f"Structure {i} in {true_file} has non-array forces: {force}")
                sys.exit(1)
            forces.append(force)

            atom_counts.append(len(atoms))
        energies = np.array(energies)
        logging.info(f"读取真实数据集 {true_file}，共 {len(structures)} 个构型。")
        return energies, forces, atom_counts
    except Exception as e:
        logging.error(f"Error reading true data from {true_file}: {e}")
        sys.exit(1)


def read_pred_data(pred_file: str) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """
    读取预测数据集，返回能量、力和原子数量。

    参数:
        pred_file (str): 预测数据集文件路径

    返回:
        Tuple[np.ndarray, List[np.ndarray], List[int]]: 能量数组，力数组列表，原子数量列表
    """
    try:
        structures = read(pred_file, index=':')
        energies = []
        forces = []
        atom_counts = []
        for i, atoms in enumerate(structures):
            # 提取预测能量
            energy = atoms.info.get('MACE_energy')
            if energy is None:
                logging.error(f"Structure {i} in {pred_file} does not have 'MACE_energy' in atoms.info.")
                sys.exit(1)
            if not isinstance(energy, (float, int)):
                logging.error(f"Structure {i} in {pred_file} has non-scalar 'MACE_energy': {energy}")
                sys.exit(1)
            energies.append(float(energy))

            # 提取预测力
            if 'MACE_forces' not in atoms.arrays:
                logging.error(f"Structure {i} in {pred_file} does not have 'MACE_forces' in atoms.arrays.")
                sys.exit(1)
            force = atoms.arrays['MACE_forces']
            if not isinstance(force, np.ndarray):
                logging.error(f"Structure {i} in {pred_file} has non-array 'MACE_forces': {force}")
                sys.exit(1)
            forces.append(force)

            # 记录原子数量
            atom_counts.append(len(atoms))
        energies = np.array(energies)
        logging.info(f"读取预测数据集 {pred_file}，共 {len(structures)} 个构型。")
        return energies, forces, atom_counts
    except Exception as e:
        logging.error(f"Error reading predicted data from {pred_file}: {e}")
        sys.exit(1)


def compute_normalized_errors(true_energies: np.ndarray,
                              pred_energies: np.ndarray,
                              true_forces: List[np.ndarray],
                              pred_forces: List[np.ndarray],
                              atom_counts: List[int]) -> Dict[str, float]:
    """
    计算归一化后的能量和力误差。

    参数:
        true_energies (np.ndarray): 真实能量
        pred_energies (np.ndarray): 预测能量
        true_forces (List[np.ndarray]): 真实力
        pred_forces (List[np.ndarray]): 预测力
        atom_counts (List[int]): 每个构型的原子数量

    返回:
        dict: 各种误差指标
    """
    # 确保输入数组的形状一致
    assert true_energies.shape == pred_energies.shape, "真实能量和预测能量的形状不一致。"
    assert len(true_forces) == len(pred_forces) == len(atom_counts), "力数据和原子数量列表长度不一致。"

    # 归一化能量为每原子能量
    true_energies_norm = true_energies / atom_counts
    pred_energies_norm = pred_energies / atom_counts
    energy_errors = pred_energies_norm - true_energies_norm
    energy_errors_meV = energy_errors * 1000  # 转换为 meV/atom

    # 计算力误差（所有力分量的误差，不进行按原子数量的归一化）
    all_force_errors = []
    for i in range(len(true_forces)):
        force_error = pred_forces[i] - true_forces[i]  # Shape: (n_atoms, 3)
        all_force_errors.extend(force_error.flatten())
    all_force_errors = np.array(all_force_errors)
    all_force_errors_meV = all_force_errors * 1000  # 转换为 meV/A

    # 计算误差指标
    rmse_energy = np.sqrt(np.mean(energy_errors_meV ** 2))
    mae_energy = np.mean(np.abs(energy_errors_meV))
    max_error_energy = np.max(np.abs(energy_errors_meV))

    rmse_force = np.sqrt(np.mean(all_force_errors_meV ** 2))
    mae_force = np.mean(np.abs(all_force_errors_meV))
    max_error_force = np.max(np.abs(all_force_errors_meV))

    # 计算相对 RMSE_F (%)
    # 真实力的均方根
    rmse_force_true = np.sqrt(np.mean([np.mean(f ** 2) for f in true_forces])) * 1000  # 转换为 meV/A
    relative_rmse_force = (rmse_force / rmse_force_true) * 100 if rmse_force_true != 0 else np.nan

    return {
        'RMSE_E': rmse_energy,              # meV/atom
        'MAE_E': mae_energy,                # meV/atom
        'Max_Error_E': max_error_energy,    # meV/atom
        'RMSE_F': rmse_force,               # meV/A
        'MAE_F': mae_force,                 # meV/A
        'Max_Error_F': max_error_force,     # meV/A
        'Relative_RMSE_F': relative_rmse_force  # %
    }


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="计算训练集、验证集和测试集的误差指标。")
    parser.add_argument('--train_true_file', type=str, required=True, help="真实训练集文件路径（train_split.extxyz）")
    parser.add_argument('--valid_true_file', type=str, required=True, help="真实验证集文件路径（valid_split.extxyz）")
    parser.add_argument('--test_true_file', type=str, required=True, help="真实测试集文件路径（test.extxyz）")
    parser.add_argument('--train_pred_file', type=str, required=True, help="预测训练集文件路径（infer_train.extxyz）")
    parser.add_argument('--valid_pred_file', type=str, required=True, help="预测验证集文件路径（infer_valid.extxyz）")
    parser.add_argument('--test_pred_file', type=str, required=True, help="预测测试集文件路径（infer_test.extxyz）")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录路径，用于保存误差结果 CSV 文件")

    args = parser.parse_args()

    # 检查文件是否存在
    required_files = [
        args.train_true_file, args.valid_true_file, args.test_true_file,
        args.train_pred_file, args.valid_pred_file, args.test_pred_file
    ]
    for file_path in required_files:
        if not os.path.isfile(file_path):
            logging.error(f"错误：文件不存在：{file_path}")
            sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取真实数据
    train_true_energies, train_true_forces, train_atom_counts = read_true_data(args.train_true_file)
    valid_true_energies, valid_true_forces, valid_atom_counts = read_true_data(args.valid_true_file)
    test_true_energies, test_true_forces, test_atom_counts = read_true_data(args.test_true_file)

    # 读取预测数据
    train_pred_energies, train_pred_forces, train_pred_atom_counts = read_pred_data(args.train_pred_file)
    valid_pred_energies, valid_pred_forces, valid_pred_atom_counts = read_pred_data(args.valid_pred_file)
    test_pred_energies, test_pred_forces, test_pred_atom_counts = read_pred_data(args.test_pred_file)

    # 确保预测和真实数据的数量一致
    if not (len(train_true_energies) == len(train_pred_energies) == len(train_atom_counts)):
        logging.error("错误：训练集真实数据和预测数据的数量不一致。")
        sys.exit(1)
    if not (len(valid_true_energies) == len(valid_pred_energies) == len(valid_atom_counts)):
        logging.error("错误：验证集真实数据和预测数据的数量不一致。")
        sys.exit(1)
    if not (len(test_true_energies) == len(test_pred_energies) == len(test_atom_counts)):
        logging.error("错误：测试集真实数据和预测数据的数量不一致。")
        sys.exit(1)

    # 计算误差
    logging.info("=== 计算训练集误差 ===")
    train_errors = compute_normalized_errors(
        train_true_energies, train_pred_energies,
        train_true_forces, train_pred_forces,
        train_atom_counts
    )

    logging.info("=== 计算验证集误差 ===")
    valid_errors = compute_normalized_errors(
        valid_true_energies, valid_pred_energies,
        valid_true_forces, valid_pred_forces,
        valid_atom_counts
    )

    logging.info("=== 计算测试集误差 ===")
    test_errors = compute_normalized_errors(
        test_true_energies, test_pred_energies,
        test_true_forces, test_pred_forces,
        test_atom_counts
    )

    # 创建 DataFrame 保存每个构型的误差
    def create_error_df(indices: List[int], atom_counts: List[int], energy_errors: Dict[str, float],
                       force_errors: Dict[str, float], dataset_type: str) -> pd.DataFrame:
        """
        创建包含误差信息的 DataFrame。

        参数:
            indices (List[int]): 构型索引列表
            atom_counts (List[int]): 原子数量列表
            energy_errors (Dict[str, float]): 能量误差字典 (meV/atom)
            force_errors (Dict[str, float]): 力的误差字典 (meV/A)
            dataset_type (str): 数据集类型（'Train', 'Validation', 'Test'）

        返回:
            pd.DataFrame: 包含误差信息的 DataFrame
        """
        df = pd.DataFrame({
            'Index': indices,
            'Atom_Count': atom_counts,
            'RMSE_E (meV/atom)': [energy_errors['RMSE_E']] * len(indices),
            'MAE_E (meV/atom)': [energy_errors['MAE_E']] * len(indices),
            'Max_Error_E (meV/atom)': [energy_errors['Max_Error_E']] * len(indices),
            'RMSE_F (meV/A)': [force_errors['RMSE_F']] * len(indices),
            'MAE_F (meV/A)': [force_errors['MAE_F']] * len(indices),
            'Max_Error_F (meV/A)': [force_errors['Max_Error_F']] * len(indices),
            'Relative_RMSE_F (%)': [force_errors['Relative_RMSE_F']] * len(indices),
            'Dataset': dataset_type
        })
        return df

    # 创建训练集 DataFrame
    train_indices = list(range(len(train_true_energies)))
    train_df = create_error_df(
        indices=train_indices,
        atom_counts=train_atom_counts,
        energy_errors=train_errors,
        force_errors=train_errors,
        dataset_type='Train'
    )

    # 创建验证集 DataFrame
    valid_indices = list(range(len(valid_true_energies)))
    valid_df = create_error_df(
        indices=valid_indices,
        atom_counts=valid_atom_counts,
        energy_errors=valid_errors,
        force_errors=valid_errors,
        dataset_type='Validation'
    )

    # 创建测试集 DataFrame
    test_indices = list(range(len(test_true_energies)))
    test_df = create_error_df(
        indices=test_indices,
        atom_counts=test_atom_counts,
        energy_errors=test_errors,
        force_errors=test_errors,
        dataset_type='Test'
    )

    # 合并所有数据集
    all_errors_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # 保存到 CSV 文件
    csv_output_path = os.path.join(args.output_dir, 'loss_and_errors.csv')
    all_errors_df.to_csv(csv_output_path, index=False)
    logging.info(f"Loss and error metrics saved to {csv_output_path}.")

    # 打印整体误差指标
    logging.info("\n=== Overall Error Metrics ===")
    for dataset_name, errors in zip(['训练集', '验证集', '测试集'], [train_errors, valid_errors, test_errors]):
        logging.info(f"{dataset_name}:")
        logging.info(
            f"  RMSE_E: {errors['RMSE_E']:.4f} meV/atom, "
            f"MAE_E: {errors['MAE_E']:.4f} meV/atom, "
            f"Max_Error_E: {errors['Max_Error_E']:.4f} meV/atom\n"
            f"  RMSE_F: {errors['RMSE_F']:.4f} meV/A, "
            f"MAE_F: {errors['MAE_F']:.4f} meV/A, "
            f"Max_Error_F: {errors['Max_Error_F']:.4f} meV/A, "
            f"Relative RMSE_F: {errors['Relative_RMSE_F']:.2f}%"
        )

if __name__ == "__main__":
    main()
