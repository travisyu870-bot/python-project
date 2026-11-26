import numpy as np

# NumPy 数值计算练习：NP-E1 至 NP-E8、NP-M1 至 NP-M4。
# 运行 python numpy_homework.py 可重现全部结果；结果没有写死。


def angle_deg(a, b):
    """NP-E3：两个同维非零向量的夹角，返回标量，单位为度。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError('a 和 b 必须是同维非空一维向量')
    a_length = np.sqrt(np.sum(a * a))
    b_length = np.sqrt(np.sum(b * b))
    if a_length == 0 or b_length == 0:
        raise ValueError('零向量没有确定的夹角')
    cosine = (a @ b) / (a_length * b_length)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def unit_vector(a):
    """NP-E5：返回同方向的单位向量，不修改输入。"""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError('a 必须是非空一维向量')
    length = np.sqrt(np.sum(a * a))
    if length == 0:
        raise ValueError('零向量不能归一化为单位向量')
    return a / length


def angles_to(query, X):
    """NP-M1：(D,) 与 (N,D) -> (N,)，零行的角度为 nan。"""
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)
    if query.ndim != 1 or query.size == 0 or X.ndim != 2 or X.shape[1] != query.size:
        raise ValueError('要求 query.shape=(D,), X.shape=(N,D)')
    query_length = np.sqrt(np.sum(query * query))
    if query_length == 0:
        raise ValueError('query 不能是零向量')
    row_lengths = np.sqrt(np.sum(X * X, axis=1))
    # where 避开零分母；out 中预先写入 nan，保留零向量的位置。
    cosine = np.full(X.shape[0], np.nan, dtype=float)
    np.divide(X @ query, row_lengths * query_length,
              out=cosine, where=row_lengths > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def top_k_cosine(query, gallery, k):
    """NP-M2：返回原 gallery 中的行索引与降序余弦分数。"""
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if (query.ndim != 1 or query.size == 0 or gallery.ndim != 2
            or gallery.shape[1] != query.size):
        raise ValueError('要求 query.shape=(D,), gallery.shape=(N,D)')
    query_length = np.sqrt(np.sum(query * query))
    if query_length == 0:
        raise ValueError('query 不能是零向量')
    row_lengths = np.sqrt(np.sum(gallery * gallery, axis=1))
    valid_rows = np.flatnonzero(row_lengths > 0)
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or k <= 0 or k > valid_rows.size):
        raise ValueError('k 必须是正整数且不超过有效行数')
    scores = (gallery[valid_rows] @ query) / (row_lengths[valid_rows] * query_length)
    scores = np.clip(scores, -1.0, 1.0)
    # 对负分数升序排序得到分数降序；stable 保留同分项的原始顺序。
    order = np.argsort(-scores, kind='stable')[:k]
    return valid_rows[order], scores[order]


def pairwise_distances(A, B):
    """NP-M3：(M,D) 与 (N,D) -> 距离 (M,N)、最近行索引 (M,)。"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1] or len(B) == 0:
        raise ValueError('A、B 必须是同特征维数的矩阵，B 不能为空')
    differences = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1))
    return distances, np.argmin(distances, axis=1)


def small_forward(X, W1, b1, W2, b2):
    """NP-M4：整批矩阵前向，返回中间数组 Z、H 和输出 Y。"""
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)
    Y = H @ W2.T + b2
    return Z, H, Y


def expect_value_error(label, function, *args):
    """运行并展示题目要求的异常边界。"""
    try:
        function(*args)
    except ValueError as error:
        print(label, 'PASS ValueError:', error)
    else:
        raise AssertionError(label + ' 应抛出 ValueError')


def main():
    np.set_printoptions(precision=6, suppress=True)

    # NP-E1：A (3,4)，第二行 (4,)，最后两列 (3,2)。
    # 边界：全非负输入经过替换保持不变；copy 避免更改原数组。
    print('\nNP-E1 数组、形状与切片')
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])
    print('ndim, shape, size:', A.ndim, A.shape, A.size)
    print('第2行:', A[1], 'shape=', A[1].shape)
    print('最后2列:\n', A[:, -2:], 'shape=', A[:, -2:].shape)
    B = A.copy()
    B[B < 0] = 0
    print('B:\n', B)
    print('原始 A:\n', A)
    print('边界：全非负输入:', np.maximum(np.array([0., 1.]), 0))

    # NP-E2：(3,)*(3,) -> (3,)；(3,)@(3,) -> 标量 shape=()。
    # 边界：零向量的长度为 0，点乘为 0。
    print('\nNP-E2 逐元素乘法、点乘与长度')
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])
    print('a*b:', a * b, 'shape=', (a * b).shape)
    print('a@b:', a @ b, 'shape=', (a @ b).shape)
    a_length = np.sqrt(np.sum(a * a))
    b_length = np.sqrt(np.sum(b * b))
    print('手算长度:', a_length, b_length)
    print('linalg 核对:', np.allclose(a_length, np.linalg.norm(a)),
          np.allclose(b_length, np.linalg.norm(b)))
    print('边界：零向量长度:', np.sqrt(np.sum(np.zeros(3) ** 2)))

    # NP-E3：(2,)、(2,) -> 角度标量；clip 防止舍入误差越出反余弦定义域。
    print('\nNP-E3 两个向量的夹角')
    print('角度:', angle_deg(np.array([1., -1.]), np.array([1., 0.])))
    expect_value_error('边界：零向量', angle_deg, np.zeros(2), np.ones(2))

    # NP-E4：(3,)-(3,) -> (3,)，两种距离均为标量。
    # 边界：两个相同点之间两种距离均为 0。
    print('\nNP-E4 两个点的距离')
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])
    difference = p - q
    print('p-q:', difference, 'shape=', difference.shape)
    print('欧氏距离:', np.sqrt(np.sum(difference ** 2)))
    print('曼哈顿距离:', np.sum(np.abs(difference)))
    print('边界：相同点:', np.linalg.norm(p - p), np.abs(p - p).sum())

    # NP-E5：(3,) -> (3,)；除法创建新数组。
    print('\nNP-E5 单位向量')
    a = np.array([-1.20, 0.40, 0.80])
    normalized = unit_vector(a)
    print('单位向量:', normalized, 'shape=', normalized.shape)
    print('长度为1:', np.allclose(np.linalg.norm(normalized), 1.0))
    print('原始 a:', a)
    expect_value_error('边界：零向量', unit_vector, np.zeros(3))

    # NP-E6：axis=0 消去行轴，对每列求和；axis=1 消去列轴，对每行求均值。
    # 边界：只有一行时 axis=0 的求和就是该行。
    print('\nNP-E6 沿不同 axis 汇总')
    A = np.array([[0.20, -0.50, 1.10], [0.80, 0.40, -0.30]])
    print('列和:', A.sum(axis=0), 'shape=', A.sum(axis=0).shape)
    print('行均值:', A.mean(axis=1), 'shape=', A.mean(axis=1).shape)
    print('整体最大值:', A.max())
    print('边界：单行列和:', A[:1].sum(axis=0))

    # NP-E7：(2,3)*(3,)+(3,) -> (2,3)，末轴3对齐，在两行间广播。
    # 边界：scale 全0时每一行都等于 bias。
    print('\nNP-E7 广播')
    X = np.array([[0.80, -0.60, 0.70], [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])
    Y = X * scale + bias
    print('Y:\n', Y, 'shape=', Y.shape)
    print('边界：零缩放:\n', X * np.zeros(3) + bias)

    # NP-E8：stack 新增批次轴；transpose 交换已有轴，不改变像素内容。
    # 边界：单张图片仍保留 batch 轴，shape=(1,3,2,2)。
    print('\nNP-E8 图像 batch')
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10
    batch = np.stack((a, b), axis=0)
    print('batch shape:', batch.shape)
    print('蓝色通道:\n', batch[:, 2], 'shape=', batch[:, 2].shape)
    print('BHWC shape:', batch.transpose(0, 2, 3, 1).shape)
    print('边界：单图 batch:', np.stack((a,), axis=0).shape)

    # NP-M1：(4,3)@(3,) -> (4,)，各行独立计算，零行返回 nan。
    print('\nNP-M1 批量夹角')
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00], [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00], [0.00, 0.00, 0.00]])
    angles = angles_to(query, X)
    print('角度:', angles, 'shape=', angles.shape)
    expect_value_error('边界：零查询', angles_to, np.zeros(3), X)

    # NP-M2：排除零行，稳定排序；相同分数保持原行顺序。
    print('\nNP-M2 余弦 Top-k')
    gallery = np.array([[0.60, -0.80, 0.00], [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00], [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(['img_a', 'img_b', 'img_c', 'img_d', 'img_zero'])
    indices, scores = top_k_cosine(query, gallery, 3)
    print('行索引:', indices, 'shape=', indices.shape)
    print('相似度:', scores)
    print('选中 ID:', ids[indices])
    expect_value_error('边界：k=0', top_k_cosine, query, gallery, 0)
    expect_value_error('边界：k超过有效行数', top_k_cosine, query, gallery, 5)
    expect_value_error('边界：k不是整数', top_k_cosine, query, gallery, 1.5)
    expect_value_error('边界：零查询', top_k_cosine, np.zeros(3), gallery, 1)

    # NP-M3：(3,1,2)-(1,2,2) -> (3,2,2)，沿末轴归约得到 (3,2)。
    # 边界：只有一个候选点时，所有最近索引都为0。
    print('\nNP-M3 距离矩阵')
    A = np.array([[0.20, -0.40], [0.80, 0.30], [-0.50, 0.70]])
    B = np.array([[0.10, 0.20], [-0.60, -0.20]])
    differences = A[:, None, :] - B[None, :, :]
    D, nearest = pairwise_distances(A, B)
    print('差数组 shape:', differences.shape)
    print('D:\n', D, 'shape=', D.shape)
    print('最近 B 行索引:', nearest)
    print('边界：单候选点:', pairwise_distances(A, B[:1])[1])

    # NP-M4：(2,3)@(3,2) -> (2,2)，两层权重均按 [out,in] 存储。
    # 边界：隐藏值恰为0时 ReLU 输出仍为0；没有样本循环。
    print('\nNP-M4 小型矩阵前向')
    X = np.array([[0.80, -0.60, 0.70], [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80], [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])
    Z, H, Y = small_forward(X, W1, b1, W2, b2)
    print('Z:\n', Z, 'shape=', Z.shape)
    print('H:\n', H, 'shape=', H.shape)
    print('Y:\n', Y, 'shape=', Y.shape)
    print('边界：ReLU 在0处:', np.maximum(np.array([-1., 0., 1.]), 0.))


if __name__ == '__main__':
    main()
