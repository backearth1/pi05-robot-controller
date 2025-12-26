# Pi05 Robot Controller

基于 OpenPI/LeRobot 的机械臂控制项目，使用 Pi05 视觉-语言-动作模型在 LIBERO 仿真环境中执行任务。

## 功能

- 支持自然语言指令控制机械臂
- 实时 MuJoCo 可视化
- 支持 LIBERO 三个任务集：
  - `libero_object`: 不同物体泛化
  - `libero_spatial`: 空间关系理解
  - `libero_goal`: 不同动作目标

## 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
cd lerobot
pip install -e ".[dev,test]"

# 下载模型
# 模型存放在 models/pi05_libero_finetuned/
```

## 使用

```bash
# 设置显示 (VNC)
export DISPLAY=:2

# 运行控制器
python run_robot.py
```

交互示例：
```
=== Pi05 Robot Controller ===

Select task suite:
1. libero_object  - 不同物体泛化
2. libero_spatial - 空间关系理解
3. libero_goal    - 不同动作目标
Enter choice (1/2/3) [default=1]: 1

> pick up the alphabet soup and place it in the basket
```

## 模型

- Pi05: 基于 PaliGemma 的视觉-语言-动作模型
- 输入: 2个相机图像 + 机械臂状态 + 语言指令
- 输出: 7维动作 (末端位置增量 + 姿态增量 + 夹爪)

## 依赖

- LeRobot
- LIBERO
- MuJoCo
- PyTorch
- robosuite
