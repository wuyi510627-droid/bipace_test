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

### 示例 4（最长轨迹）：复合任务——两个物品清洁后放不同位置

```
Your task is to: put a clean sponge in the cabinet and put a clean cloth on the diningtable.

────────────────────────────────────

步0  obs: You are in the middle of a room. Looking around, you see a cabinet 1,
          a countertop 1, a diningtable 1, a sinkbasin 1, a shelf 1, and a sidetable 1.
     act: go to countertop 1
步1  obs: You arrive at countertop 1. On the countertop 1, you see a saltshaker 1,
          a sponge 1, a bowl 2, and a cloth 1.                          (~24 token)
     act: take sponge 1 from countertop 1                               ← 拿第一个物品
步2  obs: You pick up the sponge 1 from the countertop 1.               (~10 token)
     act: go to sinkbasin 1
步3  obs: You arrive at sinkbasin 1. The sinkbasin 1 is empty.          (~8 token)
     act: clean sponge 1 with sinkbasin 1
步4  obs: You clean the sponge 1 using the sinkbasin 1.                 (~10 token)
     act: go to cabinet 1
步5  obs: You arrive at cabinet 1. The cabinet 1 is closed.             (~8 token)
     act: open cabinet 1
步6  obs: You open the cabinet 1. On the cabinet 1, you see a mug 2
          and a book 3.                                                  (~18 token)
     act: put sponge 1 in cabinet 1
步7  obs: You put the sponge 1 in the cabinet 1.                        (~10 token)
     act: close cabinet 1
步8  obs: You close the cabinet 1.                                      (~6 token)
     act: go to countertop 1                                           ← 回去拿第二个物品
步9  obs: You arrive at countertop 1. On the countertop 1, you see a saltshaker 1,
          a bowl 2, and a cloth 1.                                      (~20 token)
     act: take cloth 1 from countertop 1
步10 obs: You pick up the cloth 1 from the countertop 1.                (~10 token)
     act: go to sinkbasin 1
步11 obs: You arrive at sinkbasin 1. The sinkbasin 1 is empty.          (~8 token)
     act: clean cloth 1 with sinkbasin 1
步12 obs: You clean the cloth 1 using the sinkbasin 1.                  (~10 token)
     act: go to diningtable 1
步13 obs: You arrive at diningtable 1. On the diningtable 1, you see a
          vase 1, a candle 1, and a plate 2.                            (~20 token)
     act: put cloth 1 on diningtable 1
步14 obs: You put the cloth 1 on the diningtable 1.                     (~10 token)
     结果：成功 ✅  R = 1.0

整条轨迹 15 步，共约 190 token（纯观测）。加上历史累积后，prompt 里的 memory 远超此数。

...（实际 ALFWorld 最长复合任务可达 50 步——如"find two pencils and put one on the desk
     and one on the shelf, then put a book in the drawer"，需要穿梭于 8~10 个房间、
     在不同家具间反复移动、操作多种物品。50 步轨迹的观测本身约 500~700 token，
     但 memory 累积可达 20000+ token，这就是动机一"物理上非压不可"的直接原因。）
```

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

### 示例 4（最长轨迹）：反复搜索翻页选商品

```
指令: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
      Preferably a science or building kit.

────────────────────────────────────

步0 [搜索页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Search
act: search[science kit kids age 6-8]

────────────────────────────────────

步1 [搜索结果页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] NATIONAL GEOGRAPHIC Science Kit for Kids - 45 Experiments $29.99
     [SEP] KiwiCo Crystal Growing Kit for Kids $24.95
     [SEP] Learning Resources Gears Building Set $34.99
     [SEP] 4M Solar System Planetarium Model Kit $14.99
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Next >]

────────────────────────────────────

步2 [搜索结果第2页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Klutz LEGO Chain Reactions Craft Kit $21.99
     [SEP] Thames and Kosmos Kids First Chemistry Set $34.95
     [SEP] Snap Circuits Jr. Electronics Kit $29.95
     [SEP] Creativity for Kids Grow 'n Glow Terrarium $19.99
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Snap Circuits Jr. Electronics Kit]

────────────────────────────────────

步3 [商品详情页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Snap Circuits Jr. Electronics Kit
     [SEP] $29.95
     [SEP] Size: Standard
     [SEP] Size: Deluxe
     [SEP] Description
     [SEP] Features
     [SEP] Reviews
     [SEP] Buy Now
act: click[Reviews]

────────────────────────────────────

步4 [查看评价]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Snap Circuits Jr. Electronics Kit
     [SEP] ★★★★☆ 4.2/5 (1,247 reviews)
     [SEP] "Kids love it, but instructions are a bit confusing for younger ones" - Jane
     [SEP] "Perfect for my 7 year old grandson!" - Michael
     [SEP] "Some parts broke after a few uses" - Sarah
     [SEP] Back to Product
act: click[Back to Product]

────────────────────────────────────

步5 [回到商品页]:
act: click[Back to Search]          ← 评价一般，放弃这个

────────────────────────────────────

步6 [搜索结果第2页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Klutz LEGO Chain Reactions Craft Kit $21.99
     [SEP] Thames and Kosmos Kids First Chemistry Set $34.95
     [SEP] Snap Circuits Jr. Electronics Kit $29.95
     [SEP] Creativity for Kids Grow 'n Glow Terrarium $19.99
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Thames and Kosmos Kids First Chemistry Set]

────────────────────────────────────

步7 [商品详情页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Thames and Kosmos Kids First Chemistry Set
     [SEP] $34.95
     [SEP] Color: Blue Box
     [SEP] Color: Red Box
     [SEP] Size: Starter Set
     [SEP] Size: Deluxe Set
     [SEP] Description
     [SEP] Features
     [SEP] Reviews
     [SEP] Buy Now
act: click[Color: Blue Box]

────────────────────────────────────

步8 [选了颜色]:
obs: You have clicked Color: Blue Box.
     I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Thames and Kosmos Kids First Chemistry Set
     [SEP] $34.95
     [SEP] [clicked button] Color: Blue Box [clicked button_]
     [SEP] [button] Color: Red Box [button_]
     [SEP] [button] Size: Starter Set [button_]
     [SEP] [button] Size: Deluxe Set [button_]
     [SEP] [button] Description [button_]
     [SEP] [button] Features [button_]
     [SEP] [button] Reviews [button_]
     [SEP] [button] Buy Now [button_]
act: click[Features]

────────────────────────────────────

步9 [查看规格]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Thames and Kosmos Kids First Chemistry Set
     [SEP] $34.95
     [SEP] Ages 8+ (not suitable under 8 without adult supervision)
     [SEP] 25+ experiments included
     [SEP] All materials and instruction book included
     [SEP] Back to Product
act: click[Back to Product]         ← 年龄要求不匹配，再放弃

────────────────────────────────────

步10 [回到商品页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Thames and Kosmos Kids First Chemistry Set
     [SEP] $34.95
     [SEP] [clicked button] Color: Blue Box [clicked button_]
     [SEP] [button] Color: Red Box [button_]
     [SEP] [button] Size: Starter Set [button_]
     [SEP] [button] Size: Deluxe Set [button_]
     [SEP] [button] Description [button_]
     [SEP] [button] Features [button_]
     [SEP] [button] Reviews [button_]
     [SEP] [button] Buy Now [button_]
     [SEP] Back to Search
act: click[Back to Search]

────────────────────────────────────

步11 [搜索结果第2页]:
act: click[< Prev]

────────────────────────────────────

步12 [回到第1页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] NATIONAL GEOGRAPHIC Science Kit for Kids - 45 Experiments $29.99
     [SEP] KiwiCo Crystal Growing Kit for Kids $24.95
     [SEP] Learning Resources Gears Building Set $34.99
     [SEP] 4M Solar System Planetarium Model Kit $14.99
     [SEP] Next >
     [SEP] < Prev
     [SEP] Back to Search
act: click[Learning Resources Gears Building Set]

────────────────────────────────────

步13 [商品详情页]:
obs: I need a birthday gift for my nephew. Something educational, for ages 6-8, under $35.
     Preferably a science or building kit.
     [SEP] Learning Resources Gears Building Set
     [SEP] $34.99
     [SEP] Size: 100 Pieces
     [SEP] Size: 150 Pieces
     [SEP] Description
     [SEP] Features
     [SEP] Reviews
     [SEP] Buy Now
act: click[Size: 100 Pieces]

────────────────────────────────────

步14 [选完规格]:
act: click[Buy Now]

结果：成功 ✅  R = 1.0  (商品匹配：教育类 + 建筑套件 + 6-8岁 + <$35)
```

这条最长轨迹 15 步：两次搜索翻页、两次查看后退回、三次选规格。中途放弃两个商品后才找到对的。每步观测 200~800 token → 整条轨迹 ~8000~12000 token。

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
