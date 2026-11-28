# python项目

这份作业包括 12 道 NumPy 练习和 OpenCLIP ViT-B-32 图像编码器的六组实现。代码、运行结果和报告的使用方法如下。

代码仓库：https://github.com/travisyu870-bot/python-project

## 文件

- `numpy_homework.py`：12 道练习，包含指定函数、题目输入、结果输出、shape 解释和边界演示。
- `numpy_homework_results.txt`：实际运行练习得到的输出。
- `image_encoder.py`：完成 TODO-01 至 TODO-06 的图像编码器。
- `report.pdf`：由 LaTeX 编译的实验报告，作为提交版本。
- `report/main.tex`：报告的 LaTeX 源码，使用 XeLaTeX 编译。
- `validation.txt`：作业自带检查、额外检查及输出文件核对的实际日志。
- `verify_submission.py`：额外验证脚本，用于复现边界、切块顺序、多头注意力和新输入验证。

## 环境

Python 3.9 或以上。使用课程原附件中的 `requirements.txt` 安装依赖：

```powershell
python -m pip install -r requirements.txt
```

本次代码验证环境为 Windows、Python 3.10.8、NumPy 1.26.4、Pillow 12.3.0。主项目只使用 NumPy 和已提供的辅助模块进行前向计算，参数固定。无需 GPU 或训练。

## 单独运行 NumPy 练习

在提交文件所在文件夹运行：

```powershell
python numpy_homework.py
```

函数题保留 `angle_deg`、`unit_vector`、`angles_to`、`top_k_cosine` 名称。模块导入时不会运行示例。

## 运行主项目

1. 解压课程原始 `project1-学生版.zip`，进入其中的 `project1-numpy-student`。
2. 将本提交的 `image_encoder.py` 复制到该文件夹，替换学生版同名文件。
3. 保留原有 `helpers.py`、`preprocess_image.py`、`runtime.py`、`run.py`、`check.py`、`config.json`、`assets/` 和 `weights/`。
4. 在该文件夹执行：

```powershell
python check.py --stage all
python run.py --output output_tokens.npz
```

各阶段也可以分别检查：

```powershell
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

使用其他图片：

```powershell
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

输出文件包含按输入顺序保存的 `ids` 和 `output_tokens`；后者为有限的 float32 数组 `[B,512]`，每行长度为 1。

## 额外检查

在提交文件夹中运行，并将占位路径改为实际课程项目文件夹：

```powershell
python verify_submission.py --project-dir "C:\path\to\project1-numpy-student"
```

该脚本读取课程权重，但不读取课程参考输出。练习部分检查零向量、非法 k、稳定排序、距离和 MLP；编码器部分检查像素切块顺序、非连续数组、多头注意力的独立小型参考计算、因果掩码，以及随机生成输入的批处理一致性。课程 `check.py` 是正式的数值参考检查。

## 结果与提交范围

课程各阶段检查均通过，最终输出的最大绝对误差为 6.035e-07。逐阶段误差见报告和 `validation.txt`。不同 NumPy/BLAS 环境的舍入结果可能略有差异，使用课程给定的容差判断即可。

提交包不包含课程的大体积参数、原始参考数据和图片，主项目结合课程原附件运行。

## 编译报告

报告使用课程提供的深圳大学实验报告 LaTeX 模板，保留独立封面、五个指定章节和教师批阅页，全部采用黑字白底。将 report/ 中的文件上传至 Overleaf，选择 XeLaTeX 编译器，以 main.tex 为主文件。姓名、学号、班级及日期已填写，编译后下载 PDF 提交。
