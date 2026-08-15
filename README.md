<p align="center">
  <img src="__assets__/adagrpo_logo.png" height=150>
</p>

# GRPO-Series
This repository contains implementations of several of our GRPO projects: [Pave-GRPO](https://arxiv.org/abs/2606.01636), [AdaGRPO](https://arxiv.org/abs/2606.06828) and [G2RPO](https://arxiv.org/abs/2510.01982). 

(*Equal Contribution)(<sup>§</sup>Project Leader)(<sup>†</sup>Corresponding Author)

**[Pave-GRPO: Beyond Instantaneous Guidance through Principled Average Velocity Decomposition](https://arxiv.org/abs/2606.01636)** 
</br>
[Pengyang Ling*](https://scholar.google.com/citations?user=zO45c2QAAAAJ&hl=zh-CN),
[Jiazi Bu*](https://scholar.google.com/citations?user=a8h9Di4AAAAJ),
[Yujie Zhou*](https://scholar.google.com/citations?user=XLnXvPwAAAAJ&hl=zh-CN),
[Yibin Wang](https://codegoat24.github.io/),
[Zhenyu Hu](https://scholar.google.com/citations?user=Ju1SiBwAAAAJ&hl=en&oi=ao),
[Zihan Zhang](https://scholar.google.com/citations?user=l8fN6loAAAAJ&hl=en),
[Yi Jin](https://scholar.google.ca/citations?hl=en&user=mAJ1dCYAAAAJ),
[Huaian Chen<sup>†</sup>](https://scholar.google.com.hk/citations?hl=zh-CN&user=D6ol9XkAAAAJ),
[Yuhang Zang<sup>†</sup>](https://yuhangzang.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-2606.01636-b31b1b.svg)](https://arxiv.org/abs/2606.01636)

<details><summary>Click for the full abstract of Pave-GRPO</summary>

> Group Relative Policy Optimization(GRPO) has emerged as an effective paradigm for aligning flow-based generative models with human preferences. However, the high cost of group rollouts forces existing methods to use very few denoising steps, resulting in sparse temporal supervision and leaving most intermediate stages without direct reward guidance. To address this, we propose Pave-GRPO, which reformulates the GRPO objective through principled average velocity decomposition. Rather than generating expensive high-step rollouts, we maintain efficient few-step group sampling but decompose each coarse transition into an equivalent ensemble of finer sub-trajectories spanning multiple intermediate timesteps, propagating reward feedback to a denser set of temporal stages for more comprehensive preference alignment. Crucially, this incurs no additional stochastic rollout generation or reward evaluation: the decomposed sub-trajectories are constructed analytically around the observed transitions, requiring only a small number of extra velocity-network evaluations during the policy update. This design offers two benefits: (i) rollout-free horizon expansion: through the direct reuse of few-step group samples and their associated rewards, Pave-GRPO significantly broadens the effective optimization scope under a fixed sampling and reward budget; and (ii) comprehensive temporal supervision: by equivalently decomposing an instantaneous velocity target into a multi-timestep ensemble, it distributes reward signals across more intermediate stages of the denoising process, enabling finer-grained and more thorough preference optimization.
</details>

**[AdaGRPO: A Capability-Aware Adaptive Enhancement for Flow-based GRPO](https://arxiv.org/abs/2606.06828)** 
</br>
[Jiazi Bu*](https://scholar.google.com/citations?user=a8h9Di4AAAAJ),
[Pengyang Ling*<sup>§</sup>](https://scholar.google.com/citations?user=zO45c2QAAAAJ&hl=zh-CN),
[Yujie Zhou*](https://scholar.google.com/citations?user=XLnXvPwAAAAJ&hl=zh-CN),
[Yibin Wang](https://codegoat24.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Tianyi Wei](https://wtybest.github.io/),
[Xiaohang Zhan](https://xiaohangzhan.github.io/),
[Jiaqi Wang](https://myownskyw7.github.io/),
[Tong Wu<sup>†</sup>](https://wutong16.github.io/),
[Xingang Pan<sup>†</sup>](https://xingangpan.github.io/),
[Dahua Lin](http://dahua.site/)

[![arXiv](https://img.shields.io/badge/arXiv-2606.06828-b31b1b.svg)](https://arxiv.org/abs/2606.06828)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://bujiazi.github.io/adagrpo.github.io/)

<details><summary>Click for the full abstract of AdaGRPO</summary>

> Group Relative Policy Optimization (GRPO) has demonstrated remarkable success in aligning text-to-image (T2I) flow models with human preferences. However, we have identified that the learning loop of current flow-based GRPO is fundamentally decoupled from the learner's current capability, suffering from critical blind spots at both prompt selection and advantage estimation: (i) Existing methods sample prompts randomly, overlooking the substantial impact of data selection on reinforcement learning (RL) efficacy--a factor proven crucial in GRPO for large language models; (ii) They evaluate sample quality solely relying on intra-group statistics, lacking a global perspective to accurately measure true policy improvement. To address these issues, we propose Adaptive GRPO (AdaGRPO), a novel capability-aware RL algorithm tailored for flow models. Specifically, AdaGRPO consists of two principal components: (i) Online Curriculum Filtering Strategy: Dynamically tracks the model's proficiency and adaptively selects prompts that best match its current learning boundary; (ii) Cross-Level Advantage Fusion: Synergistically integrates fine-grained intra-group advantages with macro-level global advantages, providing a comprehensive and unbiased policy evaluation. As a lightweight, plug-and-play module, AdaGRPO can be seamlessly integrated with existing frameworks such as Flow-GRPO, DanceGRPO, and Flow-CPS. Extensive experiments demonstrate that AdaGRPO consistently drives performance gains while significantly stabilizes GRPO training for flow models.
</details>

**[[CVPR 2026] Fine-Grained GRPO for Precise Preference Alignment in Flow Models](https://arxiv.org/abs/2510.01982)** 
</br>
[Yujie Zhou*](https://github.com/YujieOuO/),
[Pengyang Ling*](https://github.com/LPengYang/),
[Jiazi Bu*](https://bujiazi.github.io/),
[Yibin Wang](https://codegoat24.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Jiaqi Wang<sup>†</sup>](https://myownskyw7.github.io/),
[Li Niu<sup>†</sup>](https://www.ustcnewly.com/),
[Guangtao Zhai](https://faculty.sjtu.edu.cn/zhaiguangtao/en/index.htm/)

[![arXiv](https://img.shields.io/badge/arXiv-2510.01982-b31b1b.svg)](https://arxiv.org/abs/2510.01982)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://bujiazi.github.io/g2rpo.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-red)](https://huggingface.co/yujieouo/G2RPO)

<details><summary>Click for the full abstract of G2RPO</summary>

> The integration of online reinforcement learning (RL) into diffusion and flow models has recently emerged as a promising approach for aligning generative models with human preferences. Stochastic sampling via Stochastic Differential Equations (SDE) is employed during the denoising process to generate diverse denoising directions for RL exploration. While existing methods effectively explore potential high-value samples, they suffer from sub-optimal preference alignment due to sparse and narrow reward signals. To address these challenges, we propose a novel **G**ranular-**GRPO** (G²RPO) framework that achieves precise and comprehensive reward assessments of sampling directions in reinforcement learning of flow models. Specifically, a *Singular Stochastic Sampling* strategy is introduced to support step-wise stochastic exploration while enforcing a high correlation between the reward and the injected noise, thereby facilitating a faithful reward for each SDE perturbation. Concurrently, to eliminate the bias inherent in fixed-granularity denoising, we introduce a *Multi-Granularity Advantage Integration* module that aggregates advantages computed at multiple diffusion scales, producing a more comprehensive and robust evaluation of the sampling directions. Experiments conducted on various reward models, including both in-domain and out-of-domain evaluations, demonstrate that our G²RPO significantly outperforms existing flow-based GRPO baselines, highlighting its effectiveness and robustness.
</details>

## 📜 News

**[2026/8/15]** The code for [Pave-GRPO](https://arxiv.org/abs/2606.01636) and [AdaGRPO](https://arxiv.org/abs/2606.06828) is released! 🚀🚀🚀

**[2026/6/5]** The [AdaGRPO](https://arxiv.org/abs/2606.06828) paper is released! 🔥🔥🔥

**[2026/6/1]** The [Pave-GRPO](https://arxiv.org/abs/2606.01636) paper is released! 🔥🔥🔥

**[2026/2/21]** [G2RPO](https://arxiv.org/abs/2510.01982) is accepted by CVPR 2026! 🎉🎉🎉 

**[2025/10/3]** The code for [G2RPO](https://arxiv.org/abs/2510.01982) is released at [here](https://github.com/bcmi/Granular-GRPO)! 🚀🚀🚀

**[2025/10/2]** The [G2RPO](https://arxiv.org/abs/2510.01982) paper is released! 🔥🔥🔥

## 🔧 Installations

### Setup repository and conda environment

```bash
git clone https://github.com/Bujiazi/GRPO-Series.git
cd GRPO-Series

conda create -n grpo python=3.10
conda activate grpo

pip install -r requirements.txt
```

It is recommended to install the pre-compiled `flash_attn`.

### Download pretrained models

```bash
bash scripts/download_pretrained_models.sh
```

Alternatively, you can download the checkpoints manually and place them following the layout below:

```
pretrained_models/
├── FLUX/                                # black-forest-labs/FLUX.1-dev
│   ├── transformer/                     #   FluxTransformer2DModel
│   ├── vae/                             #   AutoencoderKL
│   ├── text_encoder/                    #   CLIPTextModel
│   ├── text_encoder_2/                  #   T5EncoderModel
│   ├── tokenizer/                       #   CLIPTokenizer
│   └── tokenizer_2/                     #   T5TokenizerFast
├── HPSv2/                               # only required when --reward_name hpsv2
│   ├── open_clip_pytorch_model.bin      #   laion/CLIP-ViT-H-14-laion2B-s32B-b79K
│   └── HPS_v2.1_compressed.pt           #   xswu/HPSv2
└── HPSv3/                               # only required when --reward_name hpsv3
    ├── HPSv3.safetensors                #   MizzenAI/HPSv3
    ├── HPSv3_7B.yaml                    #   reward config
    └── Qwen2-VL-7B-Instruct/            #   Qwen/Qwen2-VL-7B-Instruct
```

## 🎈 Quick Start


### Pave-GRPO

```bash
bash scripts/train_pavegrpo.sh
```

### AdaGRPO

```bash
bash scripts/train_adagrpo.sh
```

### G2RPO

Coming Soon. The official implementation has been released at [here](https://github.com/bcmi/Granular-GRPO).

### Inference

```bash
python infer.py
```


## 🏗️ Todo

- [ ] Integrate the code of G2RPO (the official repo can be found [here](https://github.com/bcmi/Granular-GRPO)).
- [x] Release the code for Pave-GRPO.
- [x] Release the code for AdaGRPO.
- [x] Release the papers

## 📎 Citation 

If you find our works helpful for your research, please consider giving a star ⭐ and citation 📝 

```bibtex
@article{ling2026pave,
  title={Pave-GRPO: Beyond Instantaneous Guidance through Principled Average Velocity Decomposition},
  author={Ling, Pengyang and Bu, Jiazi and Zhou, Yujie and Wang, Yibin and Hu, Zhenyu and Zhang, Zihan and Jin, Yi and Chen, Huaian and Zang, Yuhang},
  journal={arXiv preprint arXiv:2606.01636},
  year={2026}
}

@article{bu2026adagrpo,
    title={AdaGRPO: A Capability-Aware Adaptive Enhancement for Flow-based GRPO},
    author={Bu, Jiazi and Ling, Pengyang and Zhou, Yujie and Wang, Yibin and Zang, Yuhang and Wei, Tianyi and Zhan, Xiaohang and Wang, Jiaqi and Wu, Tong and Pan, Xingang and others},
    journal={arXiv preprint arXiv:2606.06828},
    year={2026}
}

@InProceedings{Zhou_2026_CVPR,
    author    = {Zhou, Yujie and Ling, Pengyang and Bu, Jiazi and Wang, Yibin and Zang, Yuhang and Wang, Jiaqi and Niu, Li and Zhai, Guangtao},
    title     = {Fine-Grained GRPO for Precise Preference Alignment in Flow Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {20045-20054}
}
```

## 📣 Disclaimer

This is the code of Pave-GRPO, AdaGRPO and G2RPO.
All the copyrights of the demo images and audio are from community users. 
Feel free to contact us if you would like remove them.

## 💞 Acknowledgements

The code is built upon the below repositories, we thank all the contributors for open-sourcing.

* [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
* [Flow-GRPO](https://github.com/yifan123/flow_grpo)
* [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO)
* [FastVideo](https://github.com/hao-ai-lab/FastVideo)
* [DDPO](https://github.com/kvablack/ddpo-pytorch)

## 📧 Contact

Looking forward to collaborations and discussions. Please feel free to contact us via E-mail!

**Jiazi Bu**: [bujiazi001@sjtu.edu.cn](mailto:bujiazi@sjtu.edu.cn) </br>
**Pengyang Ling**: [lpyang27@mail.ustc.edu.cn](mailto:lpyang27@mail.ustc.edu.cn) </br>
**Yujie Zhou**: [yujieouo@sjtu.edu.cn](mailto:yujieouo@sjtu.edu.cn) 
