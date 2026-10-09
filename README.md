# 带方向的教学信号

这是果蝇蘑菇体里的一小块学习。多巴胺决定连接变强还是变弱，刚才发放过的肯扬细胞决定改哪几条。输出细胞谁先发放，不参与。

惩罚会把左内侧叶里、刚才发放过的肯扬细胞上已经有的突触削弱。奖赏则把这些突触加强。没有多巴胺就不改。多巴胺要是拖到气味结束后 3 秒才来，资格已经掉完了，连接也不动。左内侧叶 109 个多巴胺神经元合成一次教学，一串脉冲只乘一次。多巴胺不给细胞注入电流。

方向写在 `valence` 上，惩罚是 -1，奖赏是 +1，不是从内侧叶、垂直叶这两个名字读出来的。学习率 0.2，一次只改一小截，倍数限制在 0.05 到 2。正在发放的肯扬细胞资格接近 1，停了大约 200 毫秒就衰减掉。

## 怎么跑

先把连接表放到 `data/` 里，说明在下面。然后在这个目录执行：

```bash
pip install -r requirements.txt
python learn.py
python learn.py overlap
```

`python learn.py` 是种子 0 抽的一组气味。`overlap` 换了十个种子，看没被惩罚的气味会被拖下去多少。跑出来的数字写在 `实验结果.md`。

`learn.py` 里是学习规则。全脑以后每一步调用 `KcMbonDopamine.update`，传入这一步的肯扬细胞脉冲和多巴胺脉冲。`circuit.py` 只读连接表，分出三类细胞，这里没有神经元。`lif.py` 里的细胞用 BrainX 的 `LIFRef`：膜 20 ms，突触 5 ms，不应期 2.2 ms，步长 0.2 ms。

## 连接表

表没有放进仓库，文件太大。请自己放到：

```text
data/783_connections_processed.csv
```

用的是 FlyWire 783 的全脑连接（Dorkenwald 等，Nature，2024）。程序读进来之后，只用蘑菇体那一部分：萼、柄、内侧叶、垂直叶，左右写在脑区名字后面。列是 `pre_root_id`、`post_root_id`、`neuropil`、`syn_count`、`nt_type`、`pre_index`、`post_index`。模型里的细胞编号是 `pre_index` 和 `post_index`。
