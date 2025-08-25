1. **角度计算**
   - 计算龙头和龙身的角度值。
   - 使用数值方法求解非线性方程。
   - 将计算结果保存为 CSV 文件。

2. **速度计算**
   - 根据角度值计算龙头和龙身的速度。
   - 考虑了速度方向向量与参考方向向量的关系。
   - 将速度数据保存为 CSV 文件。

3. **位置计算**
   - 根据极径和角度计算龙头和龙身的位置信息。
   - 使用极坐标到直角坐标的转换公式。
   - 将位置信息保存为 CSV 文件。

### 极径计算
曲线的极径由以下公式定义：

\[
R(\theta) = 8.8 + \frac{0.55}{2\pi} \cdot \theta
\]

其中，\(\theta\) 表示角度（弧度制）。

### 速度分量计算
速度分量 \(v_x\) 和 \(v_y\) 的计算公式如下：

\[
v_x = -R(\theta) \cdot \sin(\theta) + R'(\theta) \cdot \cos(\theta)
\]
\[
v_y = R(\theta) \cdot \cos(\theta) + R'(\theta) \cdot \sin(\theta)
\]

其中，\(R'(\theta)\) 为极径对角度的导数。

### 非线性方程求解
本项目使用 SciPy 的 `fsolve` 函数求解以下非线性方程：

\[
g(x) = t
\]

以及：

\[
g(\theta_2) = b^2
\]

## 代码结构

### 文件结构
- `MCM-A.py`：主程序文件，包含所有功能的实现。
- `T1_angle.csv`：保存角度数据的文件。
- `T1_velocity.csv`：保存速度数据的文件。
- `T1_coordinate.csv`：保存位置信息的文件。

### 主要函数

1. `solve_theta_head(t_val)`
   - 功能：求解龙头的角度值。
   - 输入：目标值 \(t_val\)。
   - 输出：对应的角度值。

2. `next_theta(θ1, b)`
   - 功能：计算龙身的下一个角度值。
   - 输入：当前角度 \(θ1\) 和目标值 \(b\)。
   - 输出：下一个角度值。

3. `compute_velocity_next(θ_n, θ_n1, u_n)`
   - 功能：计算龙身的速度。
   - 输入：当前点和下一点的角度，以及当前点的速度大小。
   - 输出：下一点的速度大小。

4. `compute_and_save_positions()`
   - 功能：计算并保存龙头和龙身的位置信息。

## 使用方法

1. 安装必要的 Python 库：
   ```bash
   pip install numpy pandas scipy
   ```

2. 运行主程序：
   ```bash
   python MCM-A.py
   ```

3. 查看生成的 CSV 文件：
   - `T1_angle.csv`
   - `T1_velocity.csv`
   - `T1_coordinate.csv`
