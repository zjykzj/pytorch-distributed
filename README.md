<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.en.md">🇺🇸</a>
  <!-- <a title="俄语" href="../ru/README.md">🇷🇺</a> -->
</div>

 <div align="center"><a title="" href="https://github.com/zjykzj/pytorch-distributed"><img align="center" src="./imgs/DDP.png"></a></div>

<p align="center">
  «pytorch-distributed»使用<a title="" href="https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel">PyTorch DistributedDataParallel</a>实现了分布式运算，同时使用<a title="" href="https://pytorch.org/docs/stable/amp.html?highlight=amp#module-torch.cuda.amp">AMP</a>实现了混合精度运算
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

***当前仅考虑单机多卡场景***

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [使用](#使用)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

分布式运算能够充分利用多卡`GPU`算力，更快的训练得到好的模型参数；而混合精度训练一方便能够提高训练速度，还可以压缩模型大小

## 安装

通过`requirements.txt`安装运行所需依赖

```
$ pip install -r requirements.txt
```

## 使用

当前实现了`4`种训练场景：

* 单卡训练
* 多卡训练
* 单卡混合精度训练
* 多卡混合精度训练

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

* [Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
* [tczhangzhi/pytorch-distributed](https://github.com/tczhangzhi/pytorch-distributed)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjykzj/pytorch-distributed/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjykzj