# ALFWorld 与 WebShop 数据集示例

---

## ALFWorld

> 家居文本交互环境。agent 在房间里移动、拿放物品、操作家电。

### 示例 1：番茄任务

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

### 示例 4

```
Your task is to: find two pencils and put one on the desk and one on the shelf.

-- 阶段一：盲搜，找第一支铅笔（步0~7）--

步0 obs: You are in the middle of a room. Looking around, you see a desk 1,
         a shelf 1, a cabinet 1, a drawer 1, a sidetable 1, and a bed 1.
步0 act: go to desk 1
步1 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
         a mug 1, and a pen 1.
步1 act: go to drawer 1
步2 obs: You arrive at drawer 1. The drawer 1 is closed.
步2 act: open drawer 1
步3 obs: You open the drawer 1. In the drawer 1, you see a keychain 1,
         a postit 1, and a creditcard 1.
步3 act: go to cabinet 1
步4 obs: You arrive at cabinet 1. The cabinet 1 is closed.
步4 act: open cabinet 1
步5 obs: You open the cabinet 1. In the cabinet 1, you see a book 3,
         a cd 2, and a newspaper 1.
步5 act: go to sidetable 1
步6 obs: You arrive at sidetable 1. On the sidetable 1, you see an
         alarmclock 1, a tissuebox 1, and a pencil 2.
步6 act: take pencil 2 from sidetable 1                      ← 找到第一支！
步7 obs: You pick up the pencil 2 from the sidetable 1.

-- 阶段二：放第一支铅笔到 desk（步8~10）--

步7 act: go to desk 1
步8 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
         a mug 1, and a pen 1.
步8 act: put pencil 2 on desk 1
步9 obs: You put the pencil 2 on the desk 1.                  ← 第一支 ✅

-- 阶段三：搜第二支铅笔（步10~17）--

步9 act: go to bed 1
步10 obs: You arrive at bed 1. On the bed 1, you see a pillow 1,
          a book 2, and a cellphone 1.
步10 act: go to shelf 1
步11 obs: You arrive at shelf 1. On the shelf 1, you see a watch 1,
          a statue 1, and a bowl 1.
步11 act: go to sidetable 1                                   ← 折返
步12 obs: You arrive at sidetable 1. On the sidetable 1, you see an
          alarmclock 1 and a tissuebox 1.                     ← pencil 2 已拿走
步12 act: go to cabinet 1
步13 obs: You arrive at cabinet 1. The cabinet 1 is open.
          In the cabinet 1, you see a book 3, a cd 2, and a newspaper 1.
步13 act: go to drawer 1
步14 obs: You arrive at drawer 1. The drawer 1 is open.
          In the drawer 1, you see a keychain 1, a postit 1, and a creditcard 1.
步14 act: go to desk 1
步15 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
          a mug 1, a pen 1, and a pencil 2.                   ← 刚才放的，不能拿
步15 act: go to dresser 1

-- 阶段四：扩大搜索，进隔壁房间（步16~21）--

步16 obs: You arrive at dresser 1. On the dresser 1, you see a comb 1,
          a mirror 1, and a hairclip 1.
步16 act: open dresser 1
步17 obs: You open the dresser 1. In the dresser 1, you see a tshirt 2,
          a pair of socks 1, and a pencil 3.
步17 act: take pencil 3 from dresser 1                        ← 找到第二支！

-- 阶段五：放第二支铅笔到 shelf（步18~21）--

步18 obs: You pick up the pencil 3 from the dresser 1.
步18 act: go to shelf 1
步19 obs: You arrive at shelf 1. On the shelf 1, you see a watch 1,
          a statue 1, and a bowl 1.
步19 act: put pencil 3 on shelf 1
步20 obs: You put the pencil 3 on the shelf 1.                ← 第二支 ✅

-- 阶段六：确认检查（步21~27）--

步20 act: go to desk 1
步21 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
          a mug 1, a pen 1, and a pencil 2.                   ✅ desk 上有 pencil
步21 act: go to shelf 1
步22 obs: You arrive at shelf 1. On the shelf 1, you see a watch 1,
          a statue 1, a bowl 1, and a pencil 3.               ✅ shelf 上有 pencil

-- 阶段七：agent 犹豫，多余穿梭（步23~49）--
（实际 ALFWorld agent 常因不确定而继续搜索无关区域——这正是压缩要删掉的流水账）

步22 act: go to coffeetable 1
步23 obs: You arrive at coffeetable 1. On the coffeetable 1, you see a
          magazine 2, a remotecontrol 1, and a coaster 1.
步23 act: go to armchair 1
步24 obs: You arrive at armchair 1. On the armchair 1, you see a cushion 1
          and a book 1.
步24 act: go to diningtable 1
步25 obs: You arrive at diningtable 1. On the diningtable 1, you see a
          plate 2, a fork 1, a knife 1, and a cup 1.
步25 act: go to countertop 1
步26 obs: You arrive at countertop 1. On the countertop 1, you see a
          microwave 1, a toaster 1, and a cuttingboard 1.
步26 act: go to fridge 1
步27 obs: You arrive at fridge 1. The fridge 1 is closed.
步27 act: open fridge 1
步28 obs: You open the fridge 1. In the fridge 1, you see an apple 2,
          a carton of milk 1, and a butter 1.
步28 act: close fridge 1
步29 obs: You close the fridge 1.
步29 act: go to garbagecan 1
步30 obs: You arrive at garbagecan 1. The garbagecan 1 is empty.
步30 act: go to countertop 1
步31 obs: You arrive at countertop 1. On the countertop 1, you see a
          microwave 1, a toaster 1, and a cuttingboard 1.
步31 act: go to cabinet 1
步32 obs: You arrive at cabinet 1. The cabinet 1 is open.
          In the cabinet 1, you see a book 3, a cd 2, and a newspaper 1.
步32 act: go to drawer 1
步33 obs: You arrive at drawer 1. The drawer 1 is open.
          In the drawer 1, you see a keychain 1, a postit 1, and a creditcard 1.
步33 act: go to sidetable 1
步34 obs: You arrive at sidetable 1. On the sidetable 1, you see an
          alarmclock 1 and a tissuebox 1.
步34 act: go to dresser 1
步35 obs: You arrive at dresser 1. The dresser 1 is open.
          On the dresser 1, you see a comb 1, a mirror 1, and a hairclip 1.
步35 act: go to coffeetable 1
步36 obs: You arrive at coffeetable 1. On the coffeetable 1, you see a
          magazine 2, a remotecontrol 1, and a coaster 1.
步36 act: go to armchair 1
步37 obs: You arrive at armchair 1. On the armchair 1, you see a cushion 1
          and a book 1.
步37 act: go to bed 1
步38 obs: You arrive at bed 1. On the bed 1, you see a pillow 1,
          a book 2, and a cellphone 1.
步38 act: go to desk 1
步39 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
          a mug 1, a pen 1, and a pencil 2.                   ✅ 还在
步39 act: go to shelf 1
步40 obs: You arrive at shelf 1. On the shelf 1, you see a watch 1,
          a statue 1, a bowl 1, and a pencil 3.               ✅ 还在
步40 act: go to sidetable 1
步41 obs: You arrive at sidetable 1. On the sidetable 1, you see an
          alarmclock 1 and a tissuebox 1.
步41 act: go to desk 1
步42 obs: You arrive at desk 1. On the desk 1, you see a laptop 2,
          a mug 1, a pen 1, and a pencil 2.
步42 act: look around
步43 obs: You look around the room. You see a desk 1, a shelf 1,
          and a sidetable 1 nearby. Everything looks fine.

结果：成功 ✅  R = 1.0
```

> **轨迹统计**：50 步，有效步 ~18（阶段一~六），多余穿梭 22 步（阶段七）。纯观测 ~650 token，memory 累积（50 步 × ~15 token 观测 + 动作 + 系统 prompt）≈ 20000+ token。**压缩目标**：阶段七的 22 步应被压成 1~2 句摘要，节省 ~40% memory。

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

### 示例 4：反复搜索翻页选商品

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



---


