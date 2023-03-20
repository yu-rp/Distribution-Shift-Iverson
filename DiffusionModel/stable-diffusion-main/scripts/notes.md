# 参数整理
1. prompt 文字提示
14. from-file 从这个file 里 读取 prompt
11. n_samples 一个 prompt 多少张 图片，首先明白 prompt是什么 之后就是 匹配我们的方法

2. init_img 初始化图片 需要加噪声么
13. strength 噪声强度 0 到 1 ，1 就是完全噪声
8. fixed_code 是不是所有sample 都是一样的 初始化 code 这是什么

3. out_dir 输出路径
4. skip_grid 不输出grid 模式 这个要选是 或者改一下
5. skip save 不保存 这个要选否，就是不写这个参数

6. ddim_steps ddim 采样的步数 
9. ddim_eta 是不是采用确定性 采样 eta = 0.0
7. plms 是否使用 plms采样 这个不是很了解 但是可以用 [可能没法用]

10. n_iter 采样频率 不知道是什么作用
12. scale unconditional guidance的强度，当使用 0的时候 就可以得到 没有condition 的采样

13. ckpt 模型的 ckpt
14. config yaml文件
15. seed 

# 需要深入了解的参数
1. prompt 文字输入，因为我们训练的时候就是空的所以空的就行
2. from-file 什么样的file - 暂时不考虑
3. n_samples 一张图 出多少个 我们就 1就行了 之后要改
4. init_img 参数 类型
5. fixed code 没有 implementation

# 实验框架分析
1. 原有的框架包含了我们所需要的函数 我们只需要改动一下 数据的输入与存储就行了
2. batch size 改掉 这个我们要用 模型导入的batch size
3. prompt 改成动态输入 的 每个图片带一个空数据集的输入 这个可以直接从 datasets里面拿到
4. 一张图删掉 改成读数据集

# 任务
1. 添加数据读取模块 保证读取后图片的条件满足，prompt 是空数据集
2. 修改保存的模块
3. 跟新启动sh