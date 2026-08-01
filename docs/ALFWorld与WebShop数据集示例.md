# ALFWorld 与 WebShop 数据集示例

---

## ALFWorld

> 家居文本交互环境。agent 在房间里移动、拿放物品、操作家电。每条观测 4~6 个模板句，约 30~80 token。GiGPO 论文记录一条轨迹可达 50 步、超过 20k token（含历史累积）。

### 示例 1：番茄任务（我们的常用例）

```
Your task is to: heat some tomato and put it on the table.

步0 obs: You arrive at countertop 1. On the countertop 1, you see an apple 1, a bread 1, a butterknife 1, a cup 1, a fork 2, and a tomato 1.
          (~30 token)

步1 act: take tomato 1 from countertop 1
步1 obs: You pick up the tomato 1 from the countertop 1.   (~12 token)

步2 act: go to microwave 1
步2 obs: You arrive at microwave 1. The microwave 1 is closed.   (~8 token)

步3 act: open microwave 1
步3 obs: You open the microwave 1. The microwave 1 is empty.   (~8 token)

步4 act: put tomato 1 in microwave 1
步4 obs: You put the tomato 1 in the microwave 1.   (~10 token)

步5 act: heat tomato 1 with microwave
步5 obs: You heat the tomato 1 with the microwave.   (~10 token)

步6 act: take tomato 1 from microwave 1
步6 obs: You pick up the tomato 1 from the microwave 1.   (~12 token)

步7 act: go to diningtable 1
步7 obs: You arrive at diningtable 1. On the diningtable 1, you see a saltshaker 2, a peppershaker 3, a soapbottle 1, and a spatula 2.
          (~28 token)

步8 act: put tomato 1 on diningtable 1
步8 obs: You put the tomato 1 on the diningtable 1.   (~10 token)

结果：成功 ✅  R = 1.0
```

### 示例 2：闹钟任务

```
Your task is to: examine an alarmclock with the desklamp.

步0 obs: You are in the middle of a room. Looking quickly around you, you see a safe 1, a shelf 4, a sidetable 2, and a desk 1.

步1 act: go to desk 1
步1 obs: You arrive at desk 1. On the desk 1, you see a pen 1, a bowl 1, and an alarmclock 2.

步2 act: take alarmclock 2 from desk 1
步2 obs: You pick up the alarmclock 2 from the desk 1.

步3 act: go to sidetable 2
步3 obs: You arrive at sidetable 2. On the sidetable 2, you see a desklamp 1, and an alarmclock 1.

步4 act: use desklamp 1
步4 obs: You won!

结果：成功 ✅
```

### 示例 3：失败轨迹（拿错物品）

```
Your task is to: heat some tomato and put it on the table.

步0 obs: You arrive at countertop 1. On the countertop 1, you see an apple 1, a bread 1, and a tomato 1.

步1 act: take apple 1 from countertop 1          ← 拿错了！
步1 obs: You pick up the apple 1 from the countertop 1.

步2 act: go to microwave 1
步2 obs: You arrive at microwave 1. The microwave 1 is closed.

步3 act: open microwave 1
步3 obs: You open the microwave 1. The microwave 1 is empty.

步4 act: put apple 1 in microwave 1
步4 obs: You put the apple 1 in the microwave 1.

步5 act: heat apple 1 with microwave
步5 obs: You heat the apple 1 with the microwave.

...

结果：失败 ❌  R = 0.0
```

### ALFWorld 观测特点

| 特征 | 说明 |
|---|---|
| 句式 | 高度模板化（"You arrive at X. On the X, you see Y."） |
| 单步 token | 8~30 token，集中在 10~15 |
| 长度一致性 | 不同任务/不同步之间长度相对均匀（无合成钥匙-门那种长度伪影） |
| 历史累积 | 完整轨迹 50 步可达 20k+ token——**必须压缩** |
| 可验证性 | 任务有明确成功/失败判据，可写程序自动判分 |

---

## WebShop

> 模拟电商网页交互环境。agent 搜索商品、浏览结果页、选规格、加购物车。118 万真实商品，12087 条众包指令。text 模式用 `[SEP]` 拼接页面文字。每条观测约 200~800 token（取决于结果页商品数）。

### 示例 1：买一件男士防水夹克

```
指令: I want a men's lightweight packable jacket for less than $80.
      The jacket should be water-resistant and available in navy.

────────────────────────────────────

步0 [搜索页]:
obs: I want a men's lightweight packable jacket for less than $80.
     The jacket should be water-resistant and available in navy.
     [SEP] Search
act: search[lightweight packable jacket]

────────────────────────────────────

步1 [搜索结果页]:
obs: I want a men's lightweight packable jacket for less than $80.
     The jacket should be water-resistant and available in navy.
     [SEP] Amazon Essentials Men's Lightweight Water-Resistant Packable Puffer Jacket $39.99
     [SEP] Columbia Men's Watertight II Jacket $78.50
     [SEP] The North Face Men's Cyclone Jacket $85.00
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Amazon Essentials Men's Lightweight Water-Resistant Packable Puffer Jacket]

────────────────────────────────────

步2 [商品详情页]:
obs: I want a men's lightweight packable jacket for less than $80.
     The jacket should be water-resistant and available in navy.
     [SEP] Amazon Essentials Men's Lightweight Water-Resistant Packable Puffer Jacket
     [SEP] $39.99
     [SEP] Color: Black
     [SEP] Color: Navy
     [SEP] Size: M
     [SEP] Size: L
     [SEP] Size: XL
     [SEP] Description
     [SEP] Features
     [SEP] Reviews
     [SEP] Buy Now
act: click[Color: Navy]

────────────────────────────────────

步3 [选了颜色后]:
obs: You have clicked Color: Navy.
     I want a men's lightweight packable jacket for less than $80.
     The jacket should be water-resistant and available in navy.
     [SEP] Amazon Essentials Men's Lightweight Water-Resistant Packable Puffer Jacket
     [SEP] $39.99
     [SEP] [clicked button] Color: Navy [clicked button_]
     [SEP] [button] Color: Black [button_]
     [SEP] [button] Size: M [button_]
     [SEP] [button] Size: L [button_]
     [SEP] [button] Size: XL [button_]
     [SEP] [button] Description [button_]
     [SEP] [button] Features [button_]
     [SEP] [button] Reviews [button_]
     [SEP] [button] Buy Now [button_]
act: click[Size: L]

────────────────────────────────────

步4:
act: click[Buy Now]

结果：成功 ✅  R = 1.0  (商品匹配指令：防水 + Navy + <$80)
```

### 示例 2：找一双红色跑鞋

```
指令: I am looking for a pair of red running shoes for everyday training.
      It should be under $100, and size 10.

步0 [搜索]:
obs: I am looking for a pair of red running shoes for everyday training.
     It should be under $100, and size 10.
     [SEP] Search
act: search[red running shoes men]

步1 [结果页]:
obs: I am looking for a pair of red running shoes for everyday training.
     It should be under $100, and size 10.
     [SEP] Nike Revolution 6 Men's Road Running Shoes $64.99
     [SEP] ASICS Men's Gel-Contend 8 Running Shoes $59.95
     [SEP] New Balance Men's Fresh Foam Roav V1 Running Shoes $89.99
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Nike Revolution 6 Men's Road Running Shoes]

步2 [商品页]:
obs: I am looking for a pair of red running shoes for everyday training.
     It should be under $100, and size 10.
     [SEP] Nike Revolution 6 Men's Road Running Shoes
     [SEP] $64.99
     [SEP] Color: Black
     [SEP] Color: White
     [SEP] Size: 9
     [SEP] Size: 10
     [SEP] Size: 11
     [SEP] Buy Now
act: click[Color: Black]              ← 选了黑色而非红色

步3:
act: click[Size: 10]

步4:
act: click[Buy Now]

结果：失败 ❌  R = 0.0  (指令要求红色，选了黑色)
```

### 示例 3：买一个笔记本电脑支架

```
指令: Find me a laptop stand that is adjustable and under $50.

步0:
obs: Find me a laptop stand that is adjustable and under $50.
     [SEP] Search
act: search[laptop stand adjustable]

步1:
obs: Find me a laptop stand that is adjustable and under $50.
     [SEP] HUANUO Adjustable Laptop Stand, Aluminum $39.99
     [SEP] Besign LS03 Aluminum Laptop Stand $29.99
     [SEP] VIVO Adjustable Laptop Stand $44.99
     [SEP] Rain Design mStand Laptop Stand $49.90
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[HUANUO Adjustable Laptop Stand, Aluminum]

步2:
obs: Find me a laptop stand that is adjustable and under $50.
     [SEP] HUANUO Adjustable Laptop Stand, Aluminum
     [SEP] $39.99
     [SEP] Color: Silver
     [SEP] Color: Black
     [SEP] Description
     [SEP] Features
     [SEP] Reviews
     [SEP] Buy Now
act: click[Buy Now]

结果：成功 ✅  R = 1.0
```

### WebShop 观测特点

| 特征 | 说明 |
|---|---|
| 单步 token | 200~800 token，取决于结果页商品数（3~10 个商品展示） |
| 完整轨迹 | 平均 5~10 步，~2000~6000 token |
| 长度波动 | 不同查询结果数不同，观测 token 数波动较大 |
| GiGPO/BiPACE 设置 | 最大 15 步/集，历史窗口 = 2，max prompt = 4096 token |
| 自动评分 | 程序匹配——比对商品属性/类型/价格是否满足指令要求 |

---

## 两个数据集的对比

| | ALFWorld | WebShop |
|---|---|---|
| 任务类型 | 家居操作 | 电商购物 |
| 观测格式 | 模板句 "You see X." | [SEP] 拼接的页面文字 |
| 单步 token | 10~30 | 200~800 |
| 轨迹长度 | 5~50 步 | 5~15 步 |
| 完整轨迹 token | ~500~20000+ | ~2000~6000 |
| 动作空间 | 位置+物品操作（take/put/open…） | 搜索/点击/选规格/购买 |
| 成功判据 | 任务完成状态 | 商品属性匹配度 |
| 长度伪影 | ❌ 模板句长度均匀 | ⚠️ 不同页面类型长度略有不均 |
