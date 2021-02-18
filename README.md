<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/zjykzj/pytorch-distributed"><img align="center" src="./imgs/DDP.png"></a></div>

<p align="center">
  «pytorch-distributed» use <a title="" href="https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel">PyTorch DistributedDataParallel</a> implements distributed computing, and use <a title="" href="https://pytorch.org/docs/stable/amp.html?highlight=amp#module-torch.cuda.amp">AMP</a> implements the mixed precision operation
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

***At present, only single machine and multi-card scenarios are considered***

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

Distributed computing can make full use of the computing power of multi-card GPU and train better model parameters faster; At the same time, on the one hand, mixed precision training can improve the training speed, on the other hand, it can also reduce the memory occupation in the training stage and allow larger batches

## Install

```
$ pip install -r requirements.txt
```

## Usage

At present, four training scenarios are implemented:

* Single card training
* Multi-card training
* Single card hybrid precision training
* Multi-card hybrid precision training

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
* [tczhangzhi/pytorch-distributed](https://github.com/tczhangzhi/pytorch-distributed)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/pytorch-distributed/issues) or submit PRs.

Small note:

* Git submission specifications should be complied with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the[standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

[Apache License 2.0](LICENSE) © 2020 zjykzj