# coding=utf-8
import math
import argparse  # 命令行解析模块
from scipy.special import gammainc
import numpy as np
from scipy.stats import norm  # 第13个实验中导入


def get_data():
    parser = argparse.ArgumentParser(description="Randomness_Test")  # 创建解析器
    # 增加命令行参数
    parser.add_argument('data_file', type=argparse.FileType('r'), help="the sequence of bits")  # 读取数据文件
    parser.add_argument("mode", type=int, help="the choice of the tests")
    parser.add_argument("-M2", type=int, default=10, help="The length of each block  in test2")  # 第二个实验的额外参数
    parser.add_argument("-M4", type=int, default=8, help="The length of each block  in test4")  # 第四个实验的额外参数
    parser.add_argument("-M5", type=int, default=3, help="The number of rows in each matrix in test5")  # 第五个实验的额外参数
    parser.add_argument("-Q5", type=int, default=3, help="The number of columns in each matrix  in test5")
    parser.add_argument("-M7", type=int, default=10, help="The length in bits of the substring in test7")  # 第七个实验的额外参数
    parser.add_argument("-B7", type=str, default="001", help="The m-bit template in test7")
    parser.add_argument("-m7", type=int, default=3, help="The length in bits of each template in test7")
    parser.add_argument("-M8", type=int, default=10, help="The length in bits of the substring in test8")  # 第八个实验的额外参数
    parser.add_argument("-B8", type=str, default="11", help="The m-bit template in test8")
    parser.add_argument("-m8", type=int, default=2, help="The length in bits of each template in test8")
    parser.add_argument("-L9", type=int, default=2, help="The length of each block in test9")  # 第九个实验的额外参数
    parser.add_argument("-Q9", type=int, default=4, help="The number of blocks in the initialization sequence in test9")
    parser.add_argument("-M10", type=int, default=13, help="The length in bits of a block in test10")  # 第十个实验的额外参数
    parser.add_argument("-m11", type=int, default=3, help="The length of each block in test11")  # 第十一个实验的额外参数
    parser.add_argument('-m12', type=int, default=2, help="The length of each block in test12")  # 第十二个实验的额外参数
    parser.add_argument('-mode13', type=int, default=0, help="A switch for applying the test", )  # 第十三个实验的额外参数
    args = parser.parse_args()  # parse_args() 的返回值是一个命名空间，包含传递给命令的参数。该对象将参数保存其属性
    sequence = []
    sequence += [1 if c == "1" else 0 for line in args.data_file for c in line.strip() if c != " "]
    n = len(sequence)
    return sequence, n, args


# 第一个检测(频度检测)
def frequency():
    sequence, n, args = get_data()
    X = []
    Sn = 0
    for i in range(n):
        X.append(2 * sequence[i] - 1)
        Sn += X[i]
    sobs = abs(Sn) / math.sqrt(n)
    P_value = math.erfc(sobs / math.sqrt(2))
    print("第一个实验:Frequency Test,参数sequence:%s，n:%d" % (sequence, n))
    report(P_value, Sn=Sn, sobs=sobs)


# 第二个检测(块内频度检测)
def block_frequency():
    sequence, n, args = get_data()
    # args = parser.parse_args()  # 对象调用参数属性
    M = args.M2
    pi = []
    current_Value = 0
    x2 = 0
    N = int(math.floor(n / M))
    # print(type(N))
    for i in range(0, N):
        for j in range(0, M):
            if sequence[j + i * M] == 1:
                current_Value += 1
        current_Value = current_Value / M
        pi.append(current_Value)
        current_Value = 0
    #         print(pi)
    for i in range(0, N):
        x2 = x2 + (pi[i] - 1 / 2) ** 2
    x2 = 4 * M * x2
    P_value = 1 - gammainc(N / 2, x2 / 2)
    print("第二个实验:Block Frequency Test，参数sequence:%s，n:%d，M2:%d" % (sequence, n, M))
    report(P_value)


# 第三个检测(游程检测)
def run():
    sequence, n, args = get_data()
    vobs = 0
    pi = sum(sequence)
    pi = pi / n
    #         print(pi)
    t = 2 / math.sqrt(n)
    if abs(pi - 1 / 2) >= t:
        print("单比特频率测试未通过，该测试不能执行")

    for i in range(0, n - 1):
        if sequence[i] == sequence[i + 1]:
            r = 0
        elif sequence[i] != sequence[i + 1]:
            r = 1
        vobs = vobs + r
    vobs = vobs + 1
    #         print(vobs)
    a = abs(vobs - 2 * n * pi * (1 - pi))
    b = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    P_value = math.erfc(a / b)
    print("第三个实验:Run Test，参数sequence:%s，n:%d" % (sequence, n))
    report(P_value, vobs=vobs)


# 第四个检测(块内最长“1”游程检测)
def long_Run_Of_Ones():
    sequence, n, args = get_data()
    M = args.M4
    v = []
    k = 3
    x = [0.2148, 0.3672, 0.2305, 0.1875]
    x2 = 0
    vi = [0] * 4
    N = int(n / M)
    for i in range(0, N):
        max_len = 0
        current_len = 0
        for j in range(0, M):
            if sequence[j + i * M] == 1 and j + i * M < n:
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
            else:
                current_len = 0
        v.append(max_len)
    #         print(v)
    for i in v:
        if i <= 1:
            vi[0] += 1
        elif i == 2:
            vi[1] += 1
        elif i == 3:
            vi[2] += 1
        elif i >= 4:
            vi[3] += 1
    #         print(vi)
    for i in range(0, k + 1):
        x2 = x2 + (vi[i] - N * x[i]) ** 2 / (N * x[i])
    #         print(x2)
    P_value = 1 - gammainc(k / 2, x2 / 2)
    print("第四个实验:Long Run Of Ones Test,参数sequence:%s，n:%d,M4:%d" % (sequence, n, M))
    report(P_value, x2=x2)


# 第五个检测(二元矩阵秩检测)
def gfrank(data):
    b = np.array(data)
    b = b[b[:, 0].argsort()]  # 将b矩阵的第一列元素从小到大排序形成新的矩阵
    M = Q = len(b)

    b = b[::-1]
    # print(b[::-1])  #将b矩阵倒序形成新的矩阵
    l = 0
    k = 0
    while l < M:  # 按列处理
        # print(k, l)
        t = 0
        if b[k][l] == 0:
            print(b[k][l])
            for i in range(k + 1, Q):
                if b[i][l] == 1:
                    b[[k, i], :] = b[[i, k], :]  # 矩阵两行互换
                    t = t + 1
                    break
        else:
            t = t + 1
        for j in range(k + 1, Q):
            if b[j][l] == 1:
                b[j] = b[k] ^ b[j]
                t = t + 1
        if t == 0:
            k = k - 1
        # print(b)
        l = l + 1
        k = k + 1

    z = 0
    for x in range(0, M):
        if sum(b[x]) != 0:
            z = z + 1
    return (z)


def rank():
    sequence, n, args = get_data()
    M = args.M5
    Q = args.Q5
    if n >= M * Q:
        x = []
        N = int(n / (M * Q))
        c = [0] * N
        matrix_rank_list = [0] * N
        Fm = [0] * 3
        sequence = sequence[0:M * Q * N]

        for i in range(0, N):
            c[i] = sequence[M * Q * i:M * Q * (i + 1)]
            # 将列表中的字符串转换成列表形式,将列表再转换成数组形式，因为np模块中shape函数只能用于数组
            c[i] = np.array((c[i]))
            print(c[i])
            # 以M,Q为维数将数组转换成M行N列的数组
            c[i].shape = M, Q
            #             print(c[i])
            # 将数组再转成矩阵
            c[i] = np.mat(c[i])
            #             print(c[i])
            # 求每个矩阵的秩，将秩存放在a列表中
            matrix_rank_list[i] = gfrank(c[i])
            # print(a)
            # 将满秩矩阵的个数，m-1秩的个数，其他秩的个数分别存放在Fm列表中
            if matrix_rank_list[i] == M:
                Fm[0] += 1
            elif matrix_rank_list[i] == M - 1:
                Fm[1] += 1
            else:
                Fm[2] = N - Fm[0] - Fm[1]
            #         print(Fm)
        x2 = (Fm[0] - 0.2888 * N) ** 2 / (0.2888 * N) + (Fm[1] - 0.5776 * N) ** 2 / (0.5776 * N) + (
                Fm[2] - 0.1336 * N) ** 2 / (0.1336 * N)
        P_value = 1 - gammainc(1, x2 / 2)
        print("第五个实验:Rank Test，参数sequence:%s，n:%d,M5:%d,Q5:%d" % (sequence, n, M, Q))
        report(P_value, x2=x2)


# 第六个检测(离散傅里叶变换检测)
def discrete_fourier_transform():
    sequence, n, args = get_data()
    M = [0] * n
    N1 = 0
    X = [2 * i - 1 for i in sequence]
    S = np.fft.fft(X)
    for i in range(0, int(n / 2)):
        M[i] = abs(S[i])  # 计算出傅里叶变化后的复数的模
    T = math.sqrt(math.log(1 / 0.05) * n)  # 计算出T的值
    N0 = 0.95 * n / 2  # 计算出N0的值
    # print(N0)
    for i in range(0, int(n / 2)):  # 计算系数 小于门限值T 的复数个数，记作 N1 。
        if M[i] < T:
            N1 = N1 + 1
    # print(N1)
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    P_value = math.erfc(abs(d) / math.sqrt(2))
    print("第六个实验：Discrete Fourier Transform Test，参数sequence:%s，n:%d:" % (sequence, n))
    report(P_value, d=d, N0=N0, N1=N1)


# 第七个检测(非重叠模板匹配检测)
def non_overlapping_template_matching():
    sequence, n, args = get_data()
    M = args.M7
    B = args.B7
    B = [1 if i == "1" else 0 for i in B]
    m = args.m7
    N = int(n / M)
    x = [0] * N
    w = [0] * N
    x2 = 0
    #         print(N)    输出比特流被分成的块数
    for i in range(N):
        x[i] = sequence[M * i:M * (i + 1)]
        for j in range(M):
            if j + m <= M:
                if B == x[i][j:j + m]:
                    w[i] += 1
                    next_value = j + m
                else:
                    next_value = j + 1
    u = (M - m + 1) / (2 ** m)
    variance = M * (1 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))
    #         print(u,σ2)  计算均值和方差
    print(w)
    for i in range(N):
        x2 = x2 + (w[i] - u) ** 2 / variance
    P_value = 1 - gammainc(N / 2, x2 / 2)
    print("第七个实验：On Overlapping Template Matching Test，参数sequence:%s，n:%d，M7:%d，B7:%s,m7:%d" % (sequence, n, M, B, m))
    report(P_value, x2=x2)


# 第八个检测(重叠模板匹配检测)
def overlapping_template_matching():
    sequence, n, args = get_data()
    M = args.M8
    B = args.B8
    B = [1 if i == "1" else 0 for i in B]
    m = args.m8
    x2 = 0
    N = int(n / M)
    v = [0] * n
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    for i in range(N):
        a = 0
        for j in range(M):
            # 每块比较的最后一位不能超过该块数的最后一位
            if i * M + j + m <= (i + 1) * M:
                # 依次比较B与比特流
                if B == sequence[i * M + j:i * M + j + m]:
                    # 当有相应的匹配时，则a进行计数
                    a += 1
        # 将计数的a作为列表v的下标
        v[a] += 1
    v = v[:6]
    # print(v)   输出v列表的元素值
    a = (M - m + 1) / 2 ** m
    b = a / 2
    for i in range(6):
        x2 = x2 + (v[i] - N * pi[i]) ** 2 / (N * pi[i])
    P_value = 1 - gammainc(5 / 2, x2 / 2)
    print("第八个实验：Overlapping Template Matching Test，参数sequence:%s，n:%d，M8:%d，B8:%s,m8:%d" % (sequence, n, M, B, m))
    report(P_value, x2=x2)


# 第九个检测(Maurer通用统计检测)
def universal():
    sequence, n, args = get_data()
    L = args.L9
    Q = args.Q9
    N = int(n / L)
    K = N - Q
    D = [0] * Q
    T = [0] * Q
    sum = [0] * N
    sequence = [str(i) for i in sequence]
    sequence = "".join(sequence)
    InSe_sequence = sequence[:Q * L]
    TeSe_sequence = sequence[-K * L:]
    #         print(InSe_sequence,TeSe_sequence)  输出初始化的比特流，测试比特流
    A = [InSe_sequence[j * L:(j + 1) * L] for j in range(Q)]
    #          print(A[j])  在初始化比特流中输出每小块的比特流
    B = [TeSe_sequence[j * L:(j + 1) * L] for j in range(K)]
    #             print(B[j])  在测试化比特流中输出每小块的比特流
    C = A + B
    #         print(C) 将初始化和测试模块的分块比特流放在一个列表中，该模块为C

    # #将00,01,10,11分别存在T的序列中
    for i in range(Q):
        # 将十进制数字i转化为十进制，并且数字前面的0将它保留,bin(x).replace('0b',''),转化出来的是字符串类型
        D[i] = bin(i).replace('0b', '')
        # 将转化出来的字符串类型转化成整型
        D[i] = int(D[i])
        # '%02d'中的数字必须是整型，但是转化完成的又是字符串类型
        D[i] = '%02d' % (D[i])
        j = int(C[i], 2)
        T[j] = i + 1
    # print(T)   将第i块中的内容转换为十进制存放在T[j]
    # print(D)
    # 将测试化块中每个sum和求出并输出，sum列表只取测试化模块的块数
    for i in range(Q, N):
        for j in range(Q):
            if C[i] == D[j]:
                a = i + 1 - T[j]
                sum[i] = sum[i - 1] + math.log2(a)
                T[j] = i + 1
                break
    # print(sum)  # 结果[1.584962500721156, 4.169925001442312, 5.169925001442312, 5.169925001442312, 5.169925001442312, 7.169925001442312]
    # print(T)  # 输出最后的D列表中更新的数值，结果是[0, 9, 4, 10]

    # fn的值是总和sum的值除以测试化块数，输出
    fn = sum[-1] / K
    #         print(fn)  输出结果1.1949875002403854
    # 这里并没有给出L=2的均值和方差，所以按照给定的数字计算出，并输出
    P_value = math.erfc(abs((fn - 1.5374383) / (math.sqrt(2 * 1.338))))
    print("第九个实验：Maurer's Universal Statistical Test，参数sequence:%s，n:%d，L9:%d，Q9:%d" % (sequence, n, L, Q))
    report(P_value)


# 第十个检测(线性复杂度检测)
def Berlekamp_Massey_algorithm(sequence):  # 使用Berlekamp_Massey算法来计算线性复杂度Li
    N = len(sequence)
    s = sequence[:]

    for k in range(N):
        if s[k] == 1:
            break
    f = set([k + 1, 0])  # use a set to denote polynomial
    l = k + 1

    g = set([0])
    a = k
    b = 0

    for n in range(k + 1, N):
        d = 0
        for ele in f:
            d ^= s[ele + n - l]

        if d == 0:
            b += 1
        else:
            if 2 * l > n:
                f ^= set([a - b + ele for ele in g])
                b += 1
            else:
                temp = f.copy()
                f = set([b - a + ele for ele in f]) ^ g
                l = n + 1 - l
                g = temp
                a = b
                b = n - l + 1

    # output the polynomial
    def print_poly(polynomial):
        result = ''
        lis = sorted(polynomial, reverse=True)
        for i in lis:
            if i == 0:
                result += '1'
            else:
                result += 'x^%s' % str(i)

            if i != lis[-1]:
                result += ' + '
        return result

    return (print_poly(f), l)


def linear_complexity():
    sequence, n, args = get_data()
    M = args.M10
    K = 6
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    N = int(n / M)
    k = 6
    Ti = [0] * N
    Li = []
    V = [0] * 7
    chi2 = 0
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    sub_sequence = []
    for i in range(N):
        if i + 1 <= N:
            sub_sequence.append(sequence[i * M:(i + 1) * M])
    for i in range(N):
        (poly, span) = Berlekamp_Massey_algorithm(sub_sequence[i])  # 返回的是多项式和线性复杂度
        Li.append(span)

    mean = M / 2 + (9 + (-1) ** (M + 1)) / 36 - (M / 3 + 2 / 9) / 2 ** M
    for i in range(N):
        Ti[i] = ((-1) ** M) * (Li[i] - mean) + 2 / 9
        print(Ti[i])
        if Ti[i] <= -2.5:
            V[0] += 1
        elif Ti[i] > -2.5 and Ti[i] <= -1.5:
            V[1] += 1
        elif Ti[i] > -1.5 and Ti[i] <= -0.5:
            V[2] += 1
        elif Ti[i] > -0.5 and Ti[i] <= 0.5:
            V[3] += 1
        elif Ti[i] > 0.5 and Ti[i] <= 1.5:
            V[4] += 1
        elif Ti[i] > 1.5 and Ti[i] <= 2.5:
            V[5] += 1
        elif Ti[i] > 2.5:
            V[6] += 1
    print(V)
    for i in range(0, K + 1):
        chi2 = chi2 + (V[i] - N * pi[i]) * (V[i] - N * pi[i]) / (N * pi[i])
    P_value = 1 - gammainc(K / 2, chi2 / 2)
    print("第十个实验:Linear Complexity Test，参数sequence:%s，n:%d，M10:%d" % (sequence, n, M))
    report(P_value, mean=mean, chi2=chi2, N=N)


# 第十一个检测(串行测试)
def serial():
    sequence, n, args = get_data()
    m = args.m11
    sobs3 = 0
    sobs2 = 0
    sobs1 = 0
    x = [0] * 3
    sequence.extend(sequence[:m - 1])
    # print(sequence)  # 输出新的添加的序列

    # 计算出m的八进制表示
    m1 = 2 ** m  # m=3的八进制数字表示
    s1 = [0] * m1
    s11 = [0] * m1
    for i in range(0, m1):
        s1[i] = bin(i)  # 将十进制转化成二进制，输出的是字符串类型
        s1[i] = s1[i].replace('0b', '')  # 二进制转换过来的是带0b的字符串，需要将0b去掉
        s1[i] = int(s1[i])  # 将字符串转换成整型
        s1[i] = "%03d" % (s1[i])  # 转化成三位数的表示法
        s1[i] = [int(i) for i in s1[i]]
    print(s1)

    # 计算出m八进制每个数的表示，并将每个与字符串比对技术放在s11列表中
    for j in range(0, m1):  # 共有8块
        for i in range(0, n + m - 1):  # 字符串总长度
            # 因为sequence已经是列表类型，需要将s11中的每个元素也转换成列表才能进行比对
            if s1[j] == sequence[i:i + m] and i + m <= n + m:  # 不能超出比对范围
                s11[j] += 1  # 进行计数

    print(s11, m, m1)  # 输出结果是[0, 1, 1, 2, 1, 2, 2, 1]
    m2 = 2 ** (m - 1)  # m=2的八进制数字表示
    s2 = [0] * m2
    s22 = [0] * m2
    for i in range(0, m2):
        s2[i] = bin(i)  # 将十进制转化成二进制，输出的是字符串类型
        s2[i] = s2[i].replace('0b', '')  # 二进制转换过来的是带0b的字符串，需要将0b去掉
        s2[i] = int(s2[i])  # 将字符串转换成整型
        s2[i] = "%02d" % (s2[i])  # 转化成俩位数的表示法
        s2[i] = [int(i) for i in s2[i]]
    print(s2)

    for j in range(0, m2):  # 共有4块
        for i in range(0, n + m - 1 - 1):  # 字符串总长度
            # 因为sequence已经是列表类型，需要将s22中的每个元素也转换成列表才能进行比对
            if s2[j] == sequence[i:i + m - 1] and i + m - 1 <= n + m - 1 - 1:  # 不能超出比对范围
                s22[j] += 1  # 进行计数
    print(s22, m - 1, m2)  # 输出结果是[1, 3, 3, 3]

    m3 = 2 ** (m - 2)  # m=2的八进制数字表示
    s3 = [0] * m3
    s33 = [0] * m3
    for i in range(0, m3):
        s3[i] = bin(i)  # 将十进制转化成二进制，输出的是字符串类型
        s3[i] = s3[i].replace('0b', '')  # 二进制转换过来的是带0b的字符串，需要将0b去掉
        s3[i] = int(s3[i])  # 将字符串转换成整型
        s3[i] = "%01d" % (s3[i])  # 转化成俩位数的表示法
        s3[i] = [int(i) for i in s3[i]]
    print(s3)

    for j in range(0, m3):  # 共有2块
        for i in range(0, n + m - 1 - 1 - 1):  # 字符串总长度
            # s3[j]=list(s3[j])   #因为sequence已经是列表类型，需要将s22中的每个元素也转换成列表才能进行比对
            if s3[j] == sequence[i:i + m - 1 - 1] and i + m - 1 - 1 <= n + m - 1 - 1 - 1:  # 不能超出比对范围
                s33[j] += 1  # 进行计数
    print(s33, m - 2, m3)  # 输出结果是[4, 6]

    for i in range(0, 8):
        sobs3 += pow(s11[i], 2)
    x[0] = pow(2, m) / n * sobs3 - n
    # print(x[0])
    for i in range(0, 4):
        sobs2 += pow(s22[i], 2)
    x[1] = pow(2, m - 1) / n * sobs2 - n
    # print(x[1])
    for i in range(0, 2):
        sobs1 += pow(s33[i], 2)
    x[2] = pow(2, m - 2) / n * sobs1 - n
    # print(x[2])
    x2 = x[0] - x[1]
    # print(x2)
    x22 = x[0] - 2 * x[1] + x[2]
    # print(x22)
    P_value1 = 1 - gammainc(pow(2, m - 2), x2 / 2)
    P_value2 = 1 - gammainc(pow(2, m - 3), x22 / 2)
    print("第十一个实验:Serial Test，参数sequence:%s，n:%d，m11:%d" % (sequence, n, m))
    report(P_value1, P_value2, x2=x2, x22=x22)


# 第十二个检测(近似熵检测)
def approximate_entropy_test():
    sequence, n, args = get_data()
    m = args.m12
    sequence.extend(sequence[:m - 1])
    print(sequence)  # 输出添加m-1位后的序列，结果是['0', '1', '0', '0', '1', '1', '0', '1', '0', '1', '0', '1']
    print(m)
    a = 2 ** m
    x1 = [0] * a
    x11 = [0] * a
    b = 0
    C1 = [0] * a
    a1 = 0
    a2 = 0
    for i in range(a):
        x1[i] = bin(i)  # 将m是0到7之间用二进制表示，结果是字符串形式
        x1[i] = x1[i].replace('0b', '')  # 将字符串中的0b去掉
        x1[i] = int(x1[i])
        m1 = str(m)
        x1[i] = ('%0' + m1 + 'd') % (x1[i])  # 将数字整型转换成三位数的表示法
        x1[i] = [int(i) for i in x1[i]]
        # print(x1) # 输出结果，结果是[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]字符串类型

        for j in range(n + m - 1):
            if sequence[j:j + m] == x1[i] and j + m <= n + m - 1:
                b += 1
                x11[i] = b
        b = 0
        C1[i] = x11[i] / n
        # print(x11)
        # print(C1)
        if C1[i] != 0:  # 因为log函数中的参数必须大于0，所以<=0的都不符合数学定义，计算总和时要跳过所有的0值
            a1 += (math.log(C1[i])) * C1[i]
    print(a1)  # 计算结果是-1.6434177197931796
    #
    m2 = m + 1  # m2的值是4
    sequence.extend(sequence[:m2 - 1])
    # print(sequence)
    a = 2 ** m2
    x2 = [0] * a
    x22 = [0] * a
    b = 0
    C2 = [0] * a
    # print(a)
    for i in range(a):
        x2[i] = bin(i)
        x2[i] = x2[i].replace('0b', '')  # 将字符串中的0b去掉
        x2[i] = int(x2[i])
        m3 = str(m2)
        x2[i] = ('%0' + m3 + 'd') % (x2[i])
        x2[i] = [int(i) for i in x2[i]]
        # print(x2) #字符串类型
        for j in range(n + m2 - 1):
            if sequence[j:j + m2] == x2[i] and j + m2 <= n + m2 - 1:
                b += 1
                x22[i] = b
        b = 0
        C2[i] = x22[i] / n
        if C2[i] != 0:  # 因为log函数中的参数必须大于0，所以<=0的都不符合数学定义，计算总和时要跳过所有的0值
            a2 += (math.log(C2[i])) * C2[i]
    # print(a2)  # 计算结果是-1.8343719702816235

    ApEn = a1 - a2
    chi2 = 2 * n * (math.log(2) - ApEn)
    # print(ApEn,chi2)
    P_value = 1 - gammainc(2 ** (m - 1), chi2 / 2)
    print("第十二个实验:Approximate Entropy Test，参数sequence:%s，n:%d，m12:%d" % (sequence, n, m))
    report(P_value, ApEn=ApEn, chi2=chi2)


# 第十三个检测(累加和检测)
def cumulative_sums():
    sequence, n, args = get_data()
    mode = args.mode13
    n = len(sequence)
    X = [2 * b - 1 for b in sequence]
    print(X)

    S = [0] * n
    if mode == 0:
        for i in range(0, n):
            for j in range(0, i + 1):
                S[i] = S[i] + X[j]
        S = [abs(i) for i in S]
        z = max(S)
    # print(S)
    # print(z)
    if mode == 1:
        for i in range(0, n):
            for j in range(n - i - 1, n):
                S[i] = S[i] + X[j]
        S = [abs(i) for i in S]
        z = max(S)

    p1 = p2 = 0
    for i in range(int((-n / z + 1) * 4), int((n / z - 1) * 4) + 1):
        p1 = (norm().cdf((4 * i + 1) * z / math.sqrt(n)) - norm().cdf(
            (4 * i - 1) * z / math.sqrt(n))) + p1
    for i in range(int((-n / z - 3) * 4), int((n / z - 1) * 4) + 1):
        p2 = (norm().cdf((4 * i + 3) * z / math.sqrt(n)) - norm().cdf(
            (4 * i + 1) * z / math.sqrt(n))) + p2
    p_value = 1 - p1 + p2
    print("第十三个实验:Cumulative Sums，参数sequence:%s，n:%d，mode13:%d" % (sequence, n, mode))
    report(p_value, z=z)


# 第十四个检测(随机偏移检测)
def random_excursions():
    sequence, n, args = get_data()
    S = []
    X = [2 * i - 1 for i in sequence]
    # print(X)
    pi1 = [0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0312]
    pi2 = [0.75, 0.0625, 0.0469, 0.0352, 0.0264, 0.0791]
    pi3 = [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804]
    pi4 = [0.875, 0.0156, 0.0137, 0.0120, 0.0105, 0.0733]
    for i in range(n):
        X1 = X[:i + 1]
        S.append(sum(X1))
    # print(S)
    S.insert(0, 0)
    S.insert(n + 1, 0)
    # print(S)  # [0, -1, 0, 1, 0, 1, 2, 1, 2, 1, 2, 0]
    J = S.count(0) - 1
    # print(J)
    index_num = []
    for i in range(len(S)):
        if S[i] == 0:
            index_num.append(i)
    # print(index_num)  # [0, 2, 4, 11]
    cycle_list = [None] * J
    for j in range(len(index_num)):
        if j + 1 < len(index_num):
            cycle_list[j] = S[index_num[j]:index_num[j + 1] + 1]
    # print(cycle_list)
    count_num = []
    for x in range(-4, 5):
        a = 0
        if x != 0:
            count_num.append([i.count(x) for i in cycle_list])
    # print(count_num)   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 3], [0, 0, 3], [0, 0, 0], [0, 0, 0]]

    b = []
    for i in range(8):
        c = []
        for j in range(6):
            c += [count_num[i].count(j)]
        b.append(c)
    # print(b)  #[[3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0], [2, 0, 0, 1, 0, 0], [3, 0, 0, 0, 0, 0],
    # [3, 0, 0, 0, 0, 0]]

    state_x = [-4, -3, -2, -1, 1, 2, 3, 4]
    p_value = []
    for i in range(8):
        chi2 = 0
        if state_x[i] == -4:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi4[k], 2) / (J * pi4[k])
        elif state_x[i] == -3:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi3[k], 2) / (J * pi3[k])
        elif state_x[i] == -2:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi2[k], 2) / (J * pi2[k])
        elif state_x[i] == -1:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi1[k], 2) / (J * pi1[k])
        elif state_x[i] == 1:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi1[k], 2) / (J * pi1[k])
        elif state_x[i] == 2:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi2[k], 2) / (J * pi2[k])
        elif state_x[i] == 3:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi3[k], 2) / (J * pi3[k])
        elif state_x[i] == 4:
            for k in range(6):
                chi2 += pow(b[i][k] - J * pi4[k], 2) / (J * pi4[k])
        p_value.append(1 - gammainc(5 / 2, chi2 / 2))
    print("第十四个实验:Random Excursions Test，参数sequence:%s，n:%d" % (sequence, n))
    for i in range(8):
        if p_value[i] >= 0.01:
            print("当x=%d时,该序列是随机的，对应的p_value：%s" % (state_x[i], p_value[i]))
        else:
            print("当x=%d时，该序列是非随机的，对应的p_value：%s" % (state_x[i], p_value[i]))


# 第十五个检测(随机偏移检测变体)
def random_excursions_variant():
    sequence, n, args = get_data()
    S = [0] * n
    X = [2 * b - 1 for b in sequence]
    # print(X)
    for i in range(0, n):
        for j in range(0, i + 1):
            S[i] = S[i] + X[j]
    # print(S)  # 未插入0之前
    S.insert(0, 0)
    S.insert(n + 1, 0)
    # print(S)  # 插入0之后
    J = S.count(0) - 1
    # print(J)
    Si = []
    for i in range(0, n + 2):
        if S[i] == 0:
            Si.append(i)
    # print(Si)
    Circle = [0] * (J)
    for i in range(len(Si)):
        if i + 1 < len(Si):
            Circle[i] = S[Si[i]:Si[i + 1] + 1]
    print(Circle)

    S1 = set(S)  # 转换成集合
    a = list(S1)
    #         print(a)
    cc = [0] * len(a)
    p_value = [0] * len(a)
    for i in a:
        if i != 0:
            cc[i] = (S.count(i))
            b = abs(cc[i] - J)
            #                 print(b)
            c = math.sqrt(2 * J * (4 * abs(i) - 2))
            #                 print(c)
            p_value[i] = math.erfc(b / c)
    #                 print(p_value[i])
    print("第十五个实验:Random  Excursions Variant Test，参数sequence:%s，n:%d" % (sequence, n))
    for i in S1:
        if p_value[i] >= 0.01:
            print('当值为%d时，该序列是随机的，P_v值是：' % (i) + str(p_value[i]))
        else:
            print('当值为%d时，该序列是随机的，P_v值是：' % (i) + str(p_value[i]))


def report(*P_value, **kwds):
    for key, value in kwds.items():
        print("{0}:{1}".format(key, value))
    for sub_P_value in P_value:
        if sub_P_value >= 0.01:
            print("该序列是随机的,其中P_value值:" + str(sub_P_value))
        else:
            print("该序列是非随机的,其中P_value值:" + str(sub_P_value))


def main(mode):
    test_list = [frequency, block_frequency, run, long_Run_Of_Ones, rank, discrete_fourier_transform,
                 non_overlapping_template_matching, overlapping_template_matching, universal, linear_complexity, serial,
                 approximate_entropy_test,
                 cumulative_sums, random_excursions, random_excursions_variant]
    mode = mode - 1
    test_list[mode]()


if __name__ == "__main__":
    sequence, n, args = get_data()
    main(args.mode)
