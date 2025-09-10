

# CS336 Spring 2025 Assignment 2: Systems

有关作业的完整描述，请参阅作业说明文档：
[cs336_spring2025_assignment2_systems.pdf](https://grok.com/c/cs336_spring2025_assignment2_systems.pdf)

如果您发现作业说明或代码有任何问题，请随时在 GitHub 上提出 issue 或提交修复的 pull request。

## Setup

本目录的结构如下：

- [`./cs336-basics`](https://grok.com/c/cs336-basics)：包含一个名为 `cs336_basics` 的模块及其相关的 `pyproject.toml` 文件。该模块包含作业1中工作人员实现的语言模型。如果您想使用自己的实现，可以将此目录替换为您的实现。
- [`./cs336_systems`](https://grok.com/c/cs336_systems)：此文件夹基本为空！这是您将实现优化后的 Transformer 语言模型的模块。您可以从作业1（`cs336-basics`）中获取所需代码并复制过来作为起点。此外，您将在此模块中实现分布式训练和优化。

目录结构示例如下：

```sh
.
├── cs336_basics  # 名为 cs336_basics 的 Python 模块
│   ├── __init__.py
│   └── ... 作业1中的其他文件 ...
├── cs336_systems  # TODO（you）：作业2需要编写的代码
│   ├── __init__.py
│   └── ... TODO（you）：作业2所需的任何其他文件或文件夹 ...
├── README.md
├── pyproject.toml
└── ... TODO（you）：作业2所需的任何其他文件或文件夹 ...
```

如果您想使用自己的作业1实现，可以将 `cs336-basics` 目录替换为您的实现，或编辑外部的 `pyproject.toml` 文件以指向您的实现。

0. 我们使用 `uv` 管理依赖项。您可以通过运行以下命令验证 `cs336-basics` 包中的代码是否可访问：

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` 会根据 `pyproject.toml` 文件自动安装依赖项。

## Submitting

要提交作业，请运行 `./test_and_make_submission.sh`。该脚本将安装代码的依赖项，运行测试，并生成一个压缩的 tarball 文件，包含输出内容。我们应该能够解压您提交的 tarball 文件并运行 `./test_and_make_submission.sh` 来验证您的测试结果。

