import numpy as np
import pandas as pd
import random


def generate_logic_data_set(data_single_scale, Numbers, get_logic_function, channels,
							margins_cycle, duty_cycle, random_mode=False,
							gaussian=False, mu=0.0, sigma=1.0, true_table_flag=False):
	"""
	生成逻辑运算的图像数据集 (类似“MNIST”风格)，每个样本包含:
	  - 若干“通道”对应的二进制输入(0或1)，写进相应像素区域
	  - 上下左右留空白边缘
	  - 输出标签是 [0..15] 的十进制整数

	:param data_single_scale: (data_single_h, data_single_w), 整张图的总像素尺寸 (含边缘)
	:param Numbers: 数据集大小
	:param get_logic_function: 逻辑函数, 输入一个 int (0..15), 输出 int
						   超过 15 会被截断到 15
	:param channels: 通道数 (例如=4 => 输入可能是 0000~1111 共16种)
	:param margins_cycle: (top_rate, bottom_rate, left_rate, right_rate),
						  四边缘在“行列”方向上的占比
	:param duty_cycle: 在通道分配宽度中，有多少比例用来写实际像素(剩余是插槽)
	:param random_mode: 是否随机抽取输入组合 (True) 或者按顺序遍历 (False)
	:param gaussian: 是否使用高斯方式写入 1
	:param mu, sigma: 高斯写入的参数

	:return: data_set_X, data_set_Y
			 - data_set_X.shape = (Numbers, data_single_h, data_single_w)
			 - data_set_Y.shape = (Numbers,)
	"""

	def inputs_pools_logic(channels):
		"""
		构造可能的逻辑输入池 (所有二进制组合), 共 2^channels 种:
		例如 channels=4 -> 0000~1111 (共16种)

		返回一个 shape=(2^channels, channels) 的 ndarray，例:
		[
		  [0,0,0,0],
		  [0,0,0,1],
		  ...
		  [1,1,1,1]
		]
		"""
		n = 2 ** channels
		inputs_pool = []
		for i in range(n):
			# 将整数 i 的二进制表示拆分成 channels 位
			bits = [(i >> j) & 1 for j in reversed(range(channels))]
			inputs_pool.append(bits)
		return np.array(inputs_pool, dtype=np.int32)

	def pixel_write(value, pixel_scale, gaussian=False, mu=0.0, sigma=1.0):
		"""
		根据输入 value 决定如何填充一个二维像素块 (pixel_scale = (pixel_scale_h, pixel_scale_w))
		- 若 value=0: 全 0
		- 若 value=1 且 gaussian=False: 全 1
		- 若 value=1 且 gaussian=True : 生成一个 2D 高斯分布, 中心=1, 越靠边越小, 范围 [0,1]

		:param value: int, 0 或 1
		:param pixel_scale: (rows, cols), 要生成的二维数组形状
		:param gaussian: bool, True 则以高斯分布写入
		:param mu, sigma: 高斯分布相关参数(这里 mu 可以视作中心点, sigma控制扩散)
		:return: ndarray, shape = pixel_scale
		"""

		pixel_scale_w, pixel_scale_h = pixel_scale
		if value == 0:
			# 直接返回全 0
			return np.zeros((pixel_scale_h, pixel_scale_w), dtype=np.float32)

		else:
			# value == 1
			if not gaussian:
				# 全 1
				return value * np.ones((pixel_scale_h, pixel_scale_w), dtype=np.float32)
			else:
				# 2D 高斯分布 (中心最大=1, 边缘尽量接近 0)
				# 以像素中心 ( (pixel_scale_h-1)/2, (pixel_scale_w-1)/2 ) 作为高斯分布的中心
				y = np.arange(pixel_scale_h).reshape(-1, 1)
				x = np.arange(pixel_scale_w).reshape(1, -1)
				cy, cx = (pixel_scale_h - 1) / 2.0, (pixel_scale_w - 1) / 2.0

				# 计算到中心 (cy, cx) 的距离
				dist_sq = (y - cy) ** 2 + (x - cx) ** 2

				# 经典 2D 高斯: exp( - (r^2) / (2*sigma^2) )
				# 这里把 mu 可以当做中心点(也可以忽略,直接默认中心), 简化:
				gauss = np.exp(-dist_sq / (2.0 * sigma * sigma)) * value  # 这一步乘以value，应该可以实现根据写入值变化

				# 如果想让最大值严格=1，这种写法就可以
				# 也可以基于 mu 来做更多控制，这里示范一个简单实现
				return gauss.astype(np.float32)

	def get_data_single(data_single_inner, margin_T, margin_B, margin_L, margin_R):
		"""
		给不包含 margins 的 x_pixel 添加上下左右的 margin, 并返回完整图像 data_single
		:param data_single_inner: shape = (h, w) (不含边缘)
		:param margin_T: int, top
		:param margin_B: int, bottom
		:param margin_L: int, left
		:param margin_R: int, right
		:return: data_single, shape = (h+margin_T+margin_B, w+margin_L+margin_R)
		"""
		inner_h, inner_w = data_single_inner.shape
		# inner_w, inner_h = data_single_inner.shape
		out_h = inner_h + margin_T + margin_B
		out_w = inner_w + margin_L + margin_R

		data_single = np.zeros((out_h, out_w), dtype=np.float32)  # 这里应该重复定义了

		# 将 x_pixel 放置在中间
		data_single[margin_T:margin_T + inner_h, margin_L:margin_L + inner_w] = data_single_inner
		return data_single

	def data_generate(
			get_logic_function, Numbers=16,
			single_pixel_scale=(28, 28),
			single_slots_w=0,
			margin_T=0, margin_B=0, margin_L=0, margin_R=0,
			channels=4,
			random_mode=False,
			gaussian=False, mu=0.0, sigma=1.0, true_table_flag=0):
		"""
		生成整个数据集 (data_set_X, data_set_Y)

		:param Numbers: 数据量
		:param channels: 输入通道数 (如 4)
		:param get_logic_function: 一个函数 f(x_int)->y_int; 这里 x_int,y_int 范围都是 [0..15]
		:param single_pixel_scale: (h, w) 每个通道要写入的有效像素区域大小
		:param margin_*: 上下左右的边缘厚度(像素数量)
		:param random_mode: 如果 True，则从 inputs_pool_logic 里随机抽取输入; 如果 False, 则顺序遍历
		:param gaussian, mu, sigma: 用于 pixel_write 的相关控制
		:return:
		   data_set_X: shape=(Numbers, row, col)，
		   data_set_Y: shape=(Numbers,)   # 以十进制存储 [0..15]
		   x_logic_bits: 作为x标题使用，运行后不保存结果
		"""

		single_inputs_w, single_inputs_h = single_pixel_scale
		single_slots_w = single_slots_w
		pixel_scale_w = single_inputs_w  # pixel指有效区域，例如0或者1
		pixel_scale_h = single_inputs_h
		outputs_pool = 0  # 初始化一下，不然后面老是弹警告
		x_logic_bits_ls = []  # 防止后面弹警告，无意义

		# --------------------输入池获取--------------------
		if true_table_flag == 0:  # 逻辑函数输入
			# 获取所有可能的输入组合，inputs_pool.shape=(2**channels,channels)
			inputs_pool = inputs_pools_logic(channels)
		elif true_table_flag == 1:  # 真值表输入
			inputs, logic_control, outputs_pool = get_logic_function(x=0, channels=0, true_table_flag=true_table_flag)
			if len(logic_control) != 0:  # 有逻辑控制的情形
				if logic_control.shape[0] == inputs.shape[0]:
					inputs_pool = np.hstack((inputs, logic_control))
				else:
					raise ValueError("请检查真值表输入，inputs和logic_control行数不同！")
			else:  # 无逻辑控制情况
				if inputs.shape[0] != outputs_pool.shape[0]:
					raise ValueError("请检查真值表输入，inputs和outputs的行数不同！")
				inputs_pool = inputs
		elif true_table_flag == 2:  # tictac模式
			inputs_pool, outputs_pool = get_logic_function(x=0, channels=0, true_table_flag=true_table_flag,
														   Numbers=Numbers)
		elif true_table_flag == 3:  # titanic模式
			inputs_pool, outputs_pool = get_logic_function(x=0, channels=0, true_table_flag=true_table_flag,
														   Numbers=Numbers)
		else:
			inputs_pool = 0
		# --------------------输入池获取结束--------------------

		# x_logic_bits=inputs_pool
		pool_size = inputs_pool.shape[0]
		data_ls_X = []
		data_ls_y = []
		# 为了遍历 (或随机) 抽取逻辑输入
		# 如果随机: 每次随机选一个
		# 如果不随机: 循环池, 对于 Numbers > pool_size, 就让它不断重复

		idx = 0  # 用于遍历模式
		for i in range(Numbers):
			if random_mode:
				# 随机抽一个
				pick = np.random.randint(0, pool_size)
				x_logic_bits = inputs_pool[pick]
			else:
				# 顺序取
				x_logic_bits = inputs_pool[idx % pool_size]  # 这里用整除是因为考虑Numbers远大于输入池的情形，此时让其循环从输入池拿输入
				idx += 1
			# --------------------获取输出标签序号--------------------
			# 注意，这一段和前面的判断必须协同，有点屎山的味道了
			if true_table_flag == 0:  # 根据位数自动生成
				# 将 x_logic_bits 转为整数 (如 [1,0,0,1] -> 9)
				x_int = 0
				for bit in x_logic_bits:
					x_int = (x_int << 1) | bit  # 将x_int左移一位，并且按位或上bit，就是把形如[0,1,0,1]的二进制数变成十进制数
				# 应用用户提供的逻辑函数
				y_int = get_logic_function(x_int, channels, true_table_flag=true_table_flag)
				y_label_index = y_int  # y_int是逻辑函数的输出，这里把输出看作一个单分类问题，输出为y_int（例如5），也就是第y_int个标签
			elif true_table_flag == 1:  # 自定义真值表
				outpool_size = outputs_pool.shape[0]  # outputs_pool的定义前面有
				if len(outputs_pool.shape) == 1:
					y_label_index = outputs_pool[(idx - 1) % outpool_size]  # 和之前写的代码做一个兼容，如果形状为1，说明返回的输出池已经被手动截取过了
				else:
					# 这里idx-1是因为前面顺序取的时候+1了，但是此处列表指针还不应该下移
					y_label_index = np.argmax(outputs_pool[(idx - 1) % outpool_size])
			elif true_table_flag == 2:  # tictac
				outpool_size = outputs_pool.shape[0]  # outputs_pool的定义前面有
				y_label_index = outputs_pool[(idx - 1) % outpool_size]  # 和自定义真值表一样的，但是为了代码可读性，多写一段
			elif true_table_flag == 3:  # titanic
				outpool_size = outputs_pool.shape[0]  # outputs_pool的定义前面有
				y_label_index = outputs_pool[(idx - 1) % outpool_size]  # 和自定义真值表一样的，但是为了代码可读性，多写一段
			else:
				y_label_index = 0  # 防止弹警告
			x_logic_bits_ls.append(x_logic_bits)
			# --------------------获取输出标签序号结束--------------------
			x_pixel_temp_ls = []
			for bit in x_logic_bits:
				# 生成每个有效输入的矩阵（不是高斯的情况下，就是全0或全1）
				single_pixel = pixel_write(
					value=bit,
					pixel_scale=(pixel_scale_w, pixel_scale_h),
					gaussian=gaussian,
					mu=mu,
					sigma=sigma
				)

				single_slots = np.zeros((pixel_scale_h, single_slots_w), dtype=np.float32)
				x_pixel_temp_ls.append(single_pixel)
				x_pixel_temp_ls.append(single_slots)

			# 沿列方向拼接 => shape = (h, w*channels)
			data_single_inner = np.hstack(x_pixel_temp_ls)
			# 添加边缘
			data_single = get_data_single(data_single_inner, margin_T, margin_B, margin_L, margin_R)
			data_ls_X.append(data_single)
			data_ls_y.append(y_label_index)

		# 堆叠成 numpy array
		data_set_X = np.stack(data_ls_X, axis=0)  # shape=(Numbers, row_out, col_out)
		data_set_Y = np.array(data_ls_y, dtype=np.int32)  # shape=(Numbers,)
		data_set_X_code = np.array(x_logic_bits_ls)
		return data_set_X, data_set_Y, data_set_X_code

	data_single_h, data_single_w = data_single_scale
	top_rate, bottom_rate, left_rate, right_rate = margins_cycle

	margin_T = int(data_single_h * top_rate)
	margin_B = int(data_single_h * bottom_rate)
	margin_L = int(data_single_w * left_rate)
	margin_R = int(data_single_w * right_rate)

	# 剩余可用的中间高度、宽度
	inner_h = data_single_h - margin_T - margin_B
	inner_w = data_single_w - margin_L - margin_R
	single_inputs_h = inner_h  # 为变量统一命名而多写一步

	# 为每个通道划分的一段(在宽度方向)
	single_cycle_w = int(inner_w / channels)
	# 由于取整误差，下面加个偏移量给他放中心去
	remain_w = inner_w - int(inner_w / channels) * channels  # 如果不为0，也就是考虑上面取整误差
	if remain_w % 2 == 0:
		margin_L = int(margin_L + remain_w / 2)
		margin_R = int(margin_R + remain_w / 2)
	elif remain_w % 2 == 1:
		margin_L = int(margin_L + (remain_w - 1) / 2)
		margin_R = int(margin_R + (remain_w - 1) / 2 + 1)
	else:
		# 上面的int转换只是为了之后的int和float加法而存在，其实肯定是整数了
		# margin_R = margin_R + remain_w		# 旧版的直接偏移到右边
		print(f"程序肯定有点问题")

	# duty_cycle 决定其中多少部分是“有效像素区”，剩下作为“slot”空隔
	# 有效像素宽度
	single_inputs_w = int(single_cycle_w * duty_cycle)  # 这一行由于后面用的是减法定义，不需要考虑取整误差

	# slot 宽度
	single_slots_w = single_cycle_w - single_inputs_w  # (1 - duty_cycle) 的部分

	# 这里就把 single_pixel_scale 设置成: (single_inputs_w, single_inputs_h))
	single_pixel_scale = (single_inputs_w, single_inputs_h)
	print(f"有效像素面积为：行={single_inputs_h}，列={single_inputs_w}")
	# 用上述信息调用 data_generate
	data_set_X, data_set_Y, data_set_X_code = data_generate(
		Numbers=Numbers,
		channels=channels,
		get_logic_function=get_logic_function,
		single_pixel_scale=single_pixel_scale,
		single_slots_w=single_slots_w,
		margin_T=margin_T,
		margin_B=margin_B,
		margin_L=margin_L,
		margin_R=margin_R,
		random_mode=random_mode,
		gaussian=gaussian,
		mu=mu,
		sigma=sigma, true_table_flag=true_table_flag
	)

	# if x_logits_bit is None:
	# 	return data_set_X, data_set_Y
	# else:
	# 	return data_set_X, data_set_Y, x_logits_bit
	return data_set_X, data_set_Y, data_set_X_code


def generate_logic_pd_set(pd_scale, pd_numbers, margins_cycle, duty_cycle,
						  gaussian=False, mu=0.0, sigma=1.0, code_bin=True
						  ):
	"""
	生成逻辑运算的图像数据集 (类似“MNIST”风格)，每个样本包含:
	  - 若干“通道”对应的二进制输入(0或1)，写进相应像素区域
	  - 上下左右留空白边缘
	  - 输出标签是 [0..15] 的十进制整数

	:param pd_scale: (pd_h, pd_w), 整张图的总像素尺寸 (含边缘)
	:param pd_numbers: 通道数 (例如=4 => 输入可能是 0000~1111 共16种)
	:param margins_cycle: (top_rate, bottom_rate, left_rate, right_rate),
						  四边缘在“行列”方向上的占比
	:param duty_cycle: 在通道分配宽度中，有多少比例用来写实际像素(剩余是插槽)
	:param gaussian: 是否使用高斯方式写入 1
	:param mu, sigma: 高斯写入的参数

	:return: pd_set, data_set_Y
			 - pd_set.shape = (Numbers, pd_h, pd_w)
			 - data_set_Y.shape = (Numbers,)
	"""

	def inputs_pools_logic(channels, code_bin=False):
		"""
		生成指定的逻辑输入池：
		1. 仅包含单个 1 的二进制数（如 0001, 0010, 0100, 1000）
		2. 以及全 1（1111）
		3. 结果顺序：先 1111，然后单 1 组合

		:param channels: 输入的通道数
		:return: 一个 shape=(channels+1, channels) 的 ndarray
		"""
		if code_bin is True:
			# --------------------二值编码的情形--------------------
			if channels < 1:
				raise ValueError("channels 必须大于 0")
			inputs_pool = []
			# 生成全 1 的情况
			full_ones = [1] * channels

			n = 2 ** channels

			for i in range(n):
				# 将整数 i 的二进制表示拆分成 channels 位
				bits = [(i >> j) & 1 for j in reversed(range(channels))]
				inputs_pool.append(bits)

			# 组合：先 `1111`，然后 `0001` ~ `1000`
			inputs_pool = [full_ones] + inputs_pool
			return np.array(inputs_pool, dtype=np.int32)
		else:
			# --------------------独热编码的情形--------------------
			if channels < 1:
				raise ValueError("channels 必须大于 0")

			# 生成全 1 的情况
			full_ones = [1] * channels

			# 生成单 1 的所有情况
			single_ones = [[1 if j == i else 0 for j in range(channels)] for i in range(channels)]

			# 组合：先 `1111`，然后 `0001` ~ `1000`
			inputs_pool = [full_ones] + single_ones
			return np.array(inputs_pool, dtype=np.int32)

	def pixel_write(value, pixel_scale, gaussian=False, mu=0.0, sigma=1.0):
		"""
		根据输入 value 决定如何填充一个二维像素块 (pixel_scale = (single_pd_h, single_pd_w))
		- 若 value=0: 全 0
		- 若 value=1 且 gaussian=False: 全 1
		- 若 value=1 且 gaussian=True : 生成一个 2D 高斯分布, 中心=1, 越靠边越小, 范围 [0,1]

		:param value: int, 0 或 1
		:param pixel_scale: (rows, cols), 要生成的二维数组形状
		:param gaussian: bool, True 则以高斯分布写入
		:param mu, sigma: 高斯分布相关参数(这里 mu 可以视作中心点, sigma控制扩散)
		:return: ndarray, shape = pixel_scale
		"""

		pixel_scale_w, pixel_scale_h = pixel_scale
		if value == 0:
			# 直接返回全 0
			return np.zeros((pixel_scale_h, pixel_scale_w), dtype=np.float32)

		else:
			# value == 1
			if not gaussian:
				# 全 1
				return np.ones((pixel_scale_h, pixel_scale_w), dtype=np.float32)
			else:
				# 2D 高斯分布 (中心最大=1, 边缘尽量接近 0)
				# 以像素中心 ( (single_pd_h-1)/2, (single_pd_w-1)/2 ) 作为高斯分布的中心
				y = np.arange(pixel_scale_h).reshape(-1, 1)
				x = np.arange(pixel_scale_w).reshape(1, -1)
				cy, cx = (pixel_scale_h - 1) / 2.0, (pixel_scale_w - 1) / 2.0

				# 计算到中心 (cy, cx) 的距离
				dist_sq = (y - cy) ** 2 + (x - cx) ** 2

				# 经典 2D 高斯: exp( - (r^2) / (2*sigma^2) )
				# 这里把 mu 可以当做中心点(也可以忽略,直接默认中心), 简化:
				gauss = np.exp(-dist_sq / (2.0 * sigma * sigma))

				# 如果想让最大值严格=1，这种写法就可以
				# 也可以基于 mu 来做更多控制，这里示范一个简单实现
				return gauss.astype(np.float32)

	def get_data_single(data_single_inner, margin_T, margin_B, margin_L, margin_R):
		"""
		给不包含 margins 的 x_pixel 添加上下左右的 margin, 并返回完整图像 data_single
		:param data_single_inner: shape = (h, w) (不含边缘)
		:param margin_T: int, top
		:param margin_B: int, bottom
		:param margin_L: int, left
		:param margin_R: int, right
		:return: data_single, shape = (h+margin_T+margin_B, w+margin_L+margin_R)
		"""
		inner_h, inner_w = data_single_inner.shape
		# inner_w, inner_h = data_single_inner.shape
		out_h = inner_h + margin_T + margin_B
		out_w = inner_w + margin_L + margin_R

		data_single = np.zeros((out_h, out_w), dtype=np.float32)  # 这里应该重复定义了

		# 将 x_pixel 放置在中间
		data_single[margin_T:margin_T + inner_h, margin_L:margin_L + inner_w] = data_single_inner
		return data_single

	def pd_generate(
			single_pd_scale=(28, 28),
			single_slots_w=0,
			margin_T=0, margin_B=0, margin_L=0, margin_R=0,
			pd_numbers=4,
			gaussian=False, mu=0.0, sigma=1.0, code_bin=True):
		"""
		生成整个数据集 (pd_set, data_set_Y)
		:param pd_numbers: pd通道数 (如 4)
		:param single_pd_scale: (h, w) 每个通道要写入的有效像素区域大小
		:param margin_T: 上边缘厚度(像素数量)
		:param margin_B: 下边缘厚度(像素数量)
		:param margin_L: 左边缘厚度(像素数量)
		:param margin_R: 右边缘厚度(像素数量)
		:param gaussian, mu, sigma: 用于 pixel_write 的相关控制
		:return:
		   pd_set: shape=(Numbers, row, col)，
		   data_set_Y: shape=(Numbers,)   # 以十进制存储 [0..15]
		"""

		single_pd_w, single_pd_h = single_pd_scale
		single_slots_w = single_slots_w

		# 先获取所有可能的输入组合 shape=(16,pd_numbers) (以 4 路通道举例)
		inputs_pool = inputs_pools_logic(pd_numbers, code_bin=code_bin)
		pool_size = inputs_pool.shape[0]
		pd_ls = []
		# 为了遍历抽取逻辑输入
		idx = 0  # 用于遍历模式
		for i in range(pool_size):  # 二值编码
			# for i in range(2**pd_numbers + 1):  # 二值编码
			# for i in range(2**pd_numbers + 1):  # 独热编码，还有一个是叠加在一起的1111
			# 将 x_logic_bits 转为整数 (如 [1,0,0,1] -> 9)
			x_logic_bits = inputs_pool[idx % pool_size]
			idx += 1
			x_int = 0
			for bit in x_logic_bits:
				x_int = (x_int << 1) | bit

			pd_temp_ls = []
			for bit in x_logic_bits:
				# 生成每个有效输入的矩阵（不是高斯的情况下，就是全0或全1）
				single_pixel = pixel_write(
					value=bit,
					pixel_scale=(single_pd_w, single_pd_h),
					gaussian=gaussian,
					mu=mu,
					sigma=sigma
				)

				single_slots = np.zeros((single_pd_h, single_slots_w), dtype=np.float32)
				pd_temp_ls.append(single_pixel)
				pd_temp_ls.append(single_slots)

			# 沿列方向拼接 => shape = (h, w*channels)
			data_single_inner = np.hstack(pd_temp_ls)

			# 添加边缘
			data_single = get_data_single(data_single_inner, margin_T, margin_B, margin_L, margin_R)

			pd_ls.append(data_single)

		# 堆叠成 numpy array
		data_set_X = np.stack(pd_ls, axis=0)  # shape=(Numbers, row_out, col_out)

		return data_set_X

	pd_h, pd_w = pd_scale
	top_rate, bottom_rate, left_rate, right_rate = margins_cycle

	margin_T = int(pd_h * top_rate)
	margin_B = int(pd_h * bottom_rate)
	margin_L = int(pd_w * left_rate)
	margin_R = int(pd_w * right_rate)

	# 剩余可用的中间高度、宽度
	inner_h = pd_h - margin_T - margin_B
	inner_w = pd_w - margin_L - margin_R
	single_pd_h = inner_h  # 为变量统一命名而多写一步

	# 为每个通道划分的一段(在宽度方向)
	single_cycle_w = int(inner_w / pd_numbers)
	# 由于取整误差，下面加个偏移量给他放中心去
	remain_w = inner_w - int(inner_w / pd_numbers) * pd_numbers  # 如果不为0，也就是考虑上面取整误差
	if remain_w % 2 == 0:
		margin_L = int(margin_L + remain_w / 2)
		margin_R = int(margin_R + remain_w / 2)
	elif remain_w % 2 == 1:
		margin_L = int(margin_L + (remain_w - 1) / 2)
		margin_R = int(margin_R + (remain_w - 1) / 2 + 1)
	else:
		# 上面的int转换只是为了之后的int和float加法而存在，其实肯定是整数了
		# margin_R = margin_R + remain_w		# 旧版的直接偏移到右边
		print(f"程序肯定有点问题")

	# duty_cycle 决定其中多少部分是“有效像素区”，剩下作为“slot”空隔
	# 有效像素宽度
	single_pd_w = int(single_cycle_w * duty_cycle)  # 这一行由于后面用的是减法定义，不需要考虑取整误差

	# slot 宽度
	single_slots_w = single_cycle_w - single_pd_w  # (1 - duty_cycle) 的部分

	# 这里就把 single_pd_scale 设置成: (single_pd_w, single_pd_h))
	single_pd_scale = (single_pd_w, single_pd_h)
	print(f"有效像素面积为：行={single_pd_h}，列={single_pd_w}")
	# 用上述信息调用 data_generate
	pd_set = pd_generate(
		pd_numbers=pd_numbers,
		single_pd_scale=single_pd_scale,
		single_slots_w=single_slots_w,
		margin_T=margin_T,
		margin_B=margin_B,
		margin_L=margin_L,
		margin_R=margin_R,
		gaussian=gaussian,
		mu=mu,
		sigma=sigma,
		code_bin=code_bin)

	return pd_set


def save_data_set(data_set_X, data_set_Y=None, data_set_X_code=None, path=None, prefix=True, true_table_flag=0):
	"""
	将 data_set_X (形状: (Numbers, row, col)) 和 data_set_Y (形状: (Numbers,)) 以 CSV 格式保存
	- 第 0 列为标签(Y)
	- 从第 1 列开始是图像的像素(展平后的 X)

	:param data_set_X: ndarray, shape = (Numbers, row, col)或(Numbers, row * col)
	:param data_set_Y: ndarray, shape = (Numbers,)，这里记录的其实是第几个标签，而不是逻辑函数十进制输出什么的
	:param data_set_X_code: ndarray, shape = (Numbers,row,col),展示X的二进制编码，便于调试
	:param path: CSV 文件输出路径
	:param prefix: 数据是否是高斯分布生成的，决定前缀名
	"""
	import os
	if data_set_Y is not None:  # 如果存在标签
		if len(data_set_X.shape) == 3:  # 对应于数据集采用(Numbers,row,col)的组织方式
			Numbers, row, col = data_set_X.shape
			data_set_X_flat = data_set_X.reshape(Numbers, -1)  # 先将 data_set_X 展平到 (Numbers, row*col)
			# 拼成 (Numbers, 1 + row*col)，第 0 列是标签
			data_set = np.zeros((Numbers, row * col + 1), dtype=np.float32)
		elif len(data_set_X.shape) == 2:  # 对应于数据集采用(Numbers,row*col)的组织方式
			Numbers, row_col = data_set_X.shape
			data_set_X_flat = data_set_X.reshape(Numbers, -1)  # 这一步应当等价于没有操作
			# 拼成 (Numbers, 1 + row*col)，第 0 列是标签
			data_set = np.zeros((Numbers, row_col + 1), dtype=np.float32)
		else:
			print("数据维度并不是2维或3维\n本程序没有保存数据，请检查待保存的数据尺寸")
			return False
		data_set[:, 0] = data_set_Y  # 第 0 列：标签
		if data_set_X_code is not None:
			print("尚未调试结束存储二进制编码的情况，这种情况当前直接忽略")
			data_set[:, 1:] = data_set_X_flat  # 后续列：像素
		# data_set[:, 1:] = data_set_X_code  # 第1列：二进制输入编码，便于人眼调试
		# data_set[:, 2:] = data_set_X_flat  # 第2列开始的后续列：展平的像素
		else:
			data_set[:, 1:] = data_set_X_flat  # 后续列：像素
	else:  # 如果不存在标签，那么应当是在生成pd阵列，也就无所谓二进制输入编码
		if len(data_set_X.shape) == 3:  # 对应于数据集采用(Numbers,row,col)的组织方式
			Numbers, row, col = data_set_X.shape
			data_set_X_flat = data_set_X.reshape(Numbers, -1)  # 先将 data_set_X 展平到 (Numbers, row*col)
			# 拼成 (Numbers, row*col)，由于没有标签，所以第0列就是数据了
			data_set = np.zeros((Numbers, row * col), dtype=np.float32)
		elif len(data_set_X.shape) == 2:  # 对应于数据集采用(Numbers,row*col)的组织方式
			Numbers, row_col = data_set_X.shape
			data_set_X_flat = data_set_X.reshape(Numbers, -1)  # 这一步应当等价于没有操作
			# 拼成 (Numbers, row*col)，由于没有标签，所以第0列就是数据了
			data_set = np.zeros((Numbers, row_col), dtype=np.float32)
		else:
			print("数据维度并不是2维或3维\n本程序没有保存数据，请检查待保存的数据尺寸")
			return False
		# data_set[:, 0] = data_set_Y  # 第 0 列：标签
		data_set[:, :] = data_set_X_flat  # 后续列：像素

	if path is None:
		import time

		current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
		if data_set_Y is None:  # 说明是在生成pd阵列
			file_prefix = "pd"
		elif prefix is True:  # 说明是在生成数据
			file_prefix = "gaussian"
		else:
			file_prefix = "ones"
		if true_table_flag == 1:
			file_prefix = "true_table-" + file_prefix
		elif true_table_flag == 2:
			file_prefix = "tictac-" + file_prefix
		elif true_table_flag == 3:
			file_prefix = "titanic-" + file_prefix
		path = "./data_records/" + file_prefix + "-" + current_time + ".csv"
	os.makedirs(os.path.dirname(path), exist_ok=True)
	# 转成 pandas DataFrame 再保存为 CSV
	df = pd.DataFrame(data_set)
	df.to_csv(path, index=False)
	print(f"数据集已保存到: {path}")


def save_tensor(tensor):
	data_to_save = tensor.numpy()
	print(f"待保存数据尺寸：{data_to_save.shape}")

	while (True):
		op_code = input("请输入操作码：\n\t0=退出数据预处理，开始保存数据\n\t1=将其重整型为输入尺寸"
						"\n\t2=默认输入数据为复数，将其从1+j形式转换为幅度角度的1.414,4/pi形式"
						"\n\t3=默认输入数据为复数，将其从弧度制转换为角度制"
						"\n\t4=将以上全部操作一遍然后退出数据预处理")
		if op_code == "0":
			break
		elif op_code == "1":
			size = eval(input("请输入整形后的数据尺寸，输入内容必须形如「(m,n)」"))
			data_to_save = data_to_save.reshape(size)
		elif op_code == "2":
			data_to_save = np.angle(data_to_save)
		elif op_code == "3":
			data_to_save = np.degrees(data_to_save)
		elif op_code == "4":
			size = eval(input("请输入整形后的数据尺寸，输入内容必须形如「(m,n)」"))
			data_to_save = data_to_save.reshape(size)
			data_to_save = np.angle(data_to_save)
			data_to_save = np.degrees(data_to_save)
			break
		else:
			print("请检查输入操作码是否合法，合法输入仅为5种：0,1,2,3,4")
	np.savetxt("tensor_data.csv", data_to_save, delimiter=",", fmt="%.5f")  # fmt 控制小数点精度


def data_load(
		path_data="./data_records/digit-recognizer/train.csv",
		path_pd="./data_records/detector_template_28.txt", shuffle_mode=True, normalize_flag=True
):
	"""
	不再使用原先测试集 CSV 文件，而是从 train.csv 读取所有数据，
	然后通过 train_test_split 拆分为训练集 (80%) 和测试集 (20%)。
	"""
	from sklearn.model_selection import train_test_split
	# 读取训练数据（包含像素与标签）
	train = pd.read_csv(path_data, delimiter=",", engine="c",
						na_filter=False, dtype=np.float32, low_memory=False)

	# 原先 test.csv 不再读取

	# 读取探测器模板
	# pd_csv = pd.read_csv(path_pd, delimiter=" ")
	pd_csv = pd.read_csv(path_pd, dtype=np.float32, delimiter=",")

	# 提取特征和标签
	X = train.iloc[:, 1:].values.astype("float32")  # 像素值
	y = train.iloc[:, 0].values.astype("int32")  # 标签
	if shuffle_mode:
		# 按 8:2 拆分训练集与测试集
		train_X, test_X, train_y, test_y = train_test_split(
			X, y, test_size=0.2, random_state=42
		)
	else:
		train_X, test_X, train_y, test_y = train_test_split(
			X, y, test_size=0.2, shuffle=False
		)
	# 处理探测器数据
	pd_positions_all = pd_csv.iloc[0, :].values.astype("float32")
	pd_positions_single = pd_csv.iloc[1:, ].values.astype("float32")

	if normalize_flag:
		# 归一化处理	这里先注释掉了，以后记得改
		# print("归一化处理	这里先注释掉了，以后记得改")
		# train_X = train_X / 255.0
		# test_X = test_X / 255.0
		train_X, max_train_X = normalize_columns(train_X)
		test_X, max_test_X = normalize_columns(test_X)

	# 调整形状
	pd_positions_all = pd_positions_all.reshape((1, -1))

	# 返回拆分后的训练 / 测试数据，以及探测器相关数据
	return train_X, train_y, test_X, test_y, pd_positions_single, pd_positions_all


def load_data_logic(path_data=r"./data_records/gaussian-2025-03-02-16-11-20.csv"):
	'''读取训练数据（包含像素与标签）'''
	from sklearn.model_selection import train_test_split
	train = pd.read_csv(path_data, delimiter=",", engine="c",
						na_filter=False, dtype=np.float32, low_memory=False)

	# 提取特征和标签
	X = train.iloc[:, 1:].values.astype("float32")  # 像素值
	y = train.iloc[:, 0].values.astype("int32")  # 标签

	# 按 8:2 拆分训练集与测试集
	train_X, test_X, train_y, test_y = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	# 归一化处理
	train_X = train_X / 255.0
	test_X = test_X / 255.0

	return train_X, train_y, test_X, test_y, X, y


def data_load_for_kaggle_competition(path_train="./data_records/digit-recognizer/train.csv",
									 path_test="./data_records/digit-recognizer/test.csv",
									 path_pd="./data_records/detector_template_28.txt"):
	# 基本数据加载，包括探测器阵列、加载训练集和测试集

	# detector的第一行是探测器的像素位置？例如pixel0-783

	# print(detector.shape)
	# print(detector.head())

	# train 数据集的第0列是标签，例如0，6，2，第1-第784列是像素值
	train = pd.read_csv(path_train, delimiter=',', engine='c', na_filter=False, dtype=np.float32, low_memory=False)
	test = pd.read_csv(path_test, delimiter=',', engine='c', na_filter=False, dtype=np.float32, low_memory=False)
	pd_csv = pd.read_csv(path_pd, delimiter=' ')
	# m, n = train.shape
	# mt, nt = test.shape
	# print(f"train shape={m,n}")
	# print(f"test shape={mt,nt}")
	# print(train.head())

	train_x = (train.iloc[:, 1:].values).astype('float32')  # 提取训练集像素值
	train_y = train.iloc[:, 0].values.astype('int32')  # 训练集标签

	test_x = (test.iloc[:, :].values).astype('float32')  # 测试集像素值

	pd_positions_all = pd_csv.iloc[0, :].values.astype('float32')  # 所有探测器的空间分布
	pd_positions_single = pd_csv.iloc[1:, ].values.astype('float32')  # 每个探测器的空间位
	# 事实上，pd_positions_all就是np.sum(pd_positions_single,axis=0).reshape(1,-1)
	# 这一点可以从画图结果轻松看出来
	# 查看探测器位置
	# plot_pd(pd_positions_all)
	# plot_pd(pd_positions_single[0])
	# 数据归一化
	train_x = train_x / 255.0
	test_x = test_x / 255.0
	# print(pd_positions_all.shape)
	pd_positions_all = pd_positions_all.reshape((1, -1))  # 将一维数组调整为二维数组

	return train_x, train_y, test_x, pd_positions_single, pd_positions_all


def generate_tic_tac_toe_data_old(n_samples=100, save_flag=False):
	data = []
	for _ in range(n_samples):
		# 随机填充棋盘的5个关键位置
		# top_left = random.choice(['X', 'O', 'Empty'])
		# top_middle = random.choice(['X', 'O', 'Empty'])
		# middle_middle = random.choice(['X', 'O', 'Empty'])
		# bottom_left = random.choice(['X', 'O', 'Empty'])
		# bottom_right = random.choice(['X', 'O', 'Empty'])
		top_left = random.choice([0.3, 0.6, 0.9])
		top_middle = random.choice([0.3, 0.6, 0.9])
		middle_middle = random.choice([0.3, 0.6, 0.9])
		bottom_left = random.choice([0.3, 0.6, 0.9])
		bottom_right = random.choice([0.3, 0.6, 0.9])
		# 简单规则判断胜负 (如果有 3 个 X 形成线性模式，则 Win)
		win = 0  # 默认失败
		if (top_left == middle_middle == bottom_right == 0.3) or \
				(top_middle == middle_middle == bottom_left == 0.3) or \
				(middle_middle == bottom_left == bottom_right == 0.3):
			win = 1  # X 胜

		# data.append([top_left, top_middle, middle_middle, bottom_left, bottom_right, win])
		data.append([win, top_left, top_middle, middle_middle, bottom_left, bottom_right])
	datas = np.array(data)
	if save_flag:
		df = pd.DataFrame(data,
						  columns=['Win', 'Top-Left', 'Top-Middle', 'Middle-Middle', 'Bottom-Left', 'Bottom-Right'])
		file_path = "tic_tac_toe_5_features-digits.csv"
		df.to_csv(file_path, index=False)
	return datas


def generate_tic_tac_toe_data(n_samples=100, all_features_flag=True, save_flag=False):
	data = []
	if not all_features_flag:  # 仅有5个特征
		for _ in range(n_samples):
			# 随机填充棋盘的5个关键位置
			# top_left = random.choice(['X', 'O', 'Empty'])
			# top_middle = random.choice(['X', 'O', 'Empty'])
			# middle_middle = random.choice(['X', 'O', 'Empty'])
			# bottom_left = random.choice(['X', 'O', 'Empty'])
			# bottom_right = random.choice(['X', 'O', 'Empty'])
			top_left = random.choice([0.3, 0.6, 0.9])
			top_middle = random.choice([0.3, 0.6, 0.9])
			middle_middle = random.choice([0.3, 0.6, 0.9])
			bottom_left = random.choice([0.3, 0.6, 0.9])
			bottom_right = random.choice([0.3, 0.6, 0.9])
			# 简单规则判断胜负 (如果有 3 个 X 形成线性模式，则 Win)
			win = 0  # 默认失败
			win_x = 0
			win_y = 0
			if (top_left == middle_middle == bottom_right == 0.3) or \
					(top_middle == middle_middle == bottom_left == 0.3) or \
					(middle_middle == bottom_left == bottom_right == 0.3):
				win_x = 1  # X 胜
			if (top_left == middle_middle == bottom_right == 0.6) or \
					(top_middle == middle_middle == bottom_left == 0.6) or \
					(middle_middle == bottom_left == bottom_right == 0.6):
				win_y = 2  # O 胜
			if win_x and win_y:  # 这个设计其实意义不大，因为5个特征下不会出现一起赢的局面
				win = 3
			elif win_x:
				win = 1
			elif win_y:
				win = 2
			else:
				win = 0  # 其实重复赋值了，为了美观写一下
			# data.append([top_left, top_middle, middle_middle, bottom_left, bottom_right, win])
			data.append([win, top_left, top_middle, middle_middle, bottom_left, bottom_right])
		tic_tac_datas = np.array(data)
		if save_flag:
			df = pd.DataFrame(data,
							  columns=['Win', 'Top-Left', 'Top-Middle', 'Middle-Middle', 'Bottom-Left', 'Bottom-Right'])
			file_path = "tic_tac_toe_5_features-digits.csv"
			df.to_csv(file_path, index=False)
	else:  # 完整的9个特征
		for _ in range(n_samples):
			# 随机填充棋盘的5个关键位置
			# top_left = random.choice(['X', 'O', 'Empty'])
			# top_middle = random.choice(['X', 'O', 'Empty'])
			# middle_middle = random.choice(['X', 'O', 'Empty'])
			# bottom_left = random.choice(['X', 'O', 'Empty'])
			# bottom_right = random.choice(['X', 'O', 'Empty'])
			# top_left = random.choice([0.3, 0.6, 0.9])
			# top_middle = random.choice([0.3, 0.6, 0.9])
			# top_right = random.choice([0.3, 0.6, 0.9])
			# middle_left = random.choice([0.3, 0.6, 0.9])
			# middle_middle = random.choice([0.3, 0.6, 0.9])
			# middle_right = random.choice([0.3, 0.6, 0.9])
			# bottom_left = random.choice([0.3, 0.6, 0.9])
			# bottom_middle = random.choice([0.3, 0.6, 0.9])
			# bottom_right = random.choice([0.3, 0.6, 0.9])

			# top_left = random.choice([0.3, 0.6, 0])
			# top_middle = random.choice([0.3, 0.6, 0])
			# top_right = random.choice([0.3, 0.6, 0])
			# middle_left = random.choice([0.3, 0.6, 0])
			# middle_middle = random.choice([0.3, 0.6, 0])
			# middle_right = random.choice([0.3, 0.6, 0])
			# bottom_left = random.choice([0.3, 0.6, 0])
			# bottom_middle = random.choice([0.3, 0.6, 0])
			# bottom_right = random.choice([0.3, 0.6, 0])

			top_left = random.choice([0.3, 0, 0])
			top_middle = random.choice([0.3, 0, 0])
			top_right = random.choice([0.3, 0, 0])
			middle_left = random.choice([0.3, 0, 0])
			middle_middle = random.choice([0.3, 0, 0])
			middle_right = random.choice([0.3, 0, 0])
			bottom_left = random.choice([0.3, 0, 0])
			bottom_middle = random.choice([0.3, 0, 0])
			bottom_right = random.choice([0.3, 0, 0])
			# 简单规则判断胜负 (如果有 3 个 X 形成线性模式，则 Win)
			win = 0  # 默认失败
			# if (top_left == middle_middle == bottom_right == 0.3) or \
			# 		(top_middle == middle_middle == bottom_left == 0.3) or \
			# 		(middle_middle == bottom_left == bottom_right == 0.3):
			# 	win = 1  # X 胜
			win_x = 0
			win_y = 0
			if (top_left == top_middle == top_right == 0.3) or \
					(top_left == middle_middle == bottom_right == 0.3) or \
					(top_left == top_middle == top_right == 0.3) or \
					(top_left == middle_left == bottom_left == 0.3) or \
					(top_middle == middle_middle == bottom_middle == 0.3) or \
					(top_right == middle_right == bottom_right == 0.3) or \
					(middle_left == middle_middle == middle_right == 0.3) or \
					(bottom_left == bottom_middle == bottom_right == 0.3) or \
					(bottom_left == middle_middle == top_right == 0.3):
				win_x = 1
			if (top_left == top_middle == top_right == 0.6) or \
					(top_left == middle_middle == bottom_right == 0.6) or \
					(top_left == top_middle == top_right == 0.6) or \
					(top_left == middle_left == bottom_left == 0.6) or \
					(top_middle == middle_middle == bottom_middle == 0.6) or \
					(top_right == middle_right == bottom_right == 0.6) or \
					(middle_left == middle_middle == middle_right == 0.6) or \
					(bottom_left == bottom_middle == bottom_right == 0.6) or \
					(bottom_left == middle_middle == top_right == 0.6):
				win_y = 1
			if win_x and win_y:
				win = 3
			elif win_x:
				win = 1
			elif win_y:
				win = 2
			else:
				win = 0  # 其实重复赋值了，为了美观写一下
			# data.append([top_left, top_middle, middle_middle, bottom_left, bottom_right, win])
			data.append([win, top_left, top_middle, top_right,
						 middle_left, middle_middle, middle_right,
						 bottom_left, bottom_middle, bottom_right])
		tic_tac_datas = np.array(data)
		if save_flag:
			df = pd.DataFrame(data, columns=
			['Win',
			 'Top-Left', 'Top-Middle', 'Top-Right',
			 "Middle-Left", 'Middle-Middle', "Middle-Right",
			 'Bottom-Left', "Bottom-Middle", 'Bottom-Right'])
			file_path = "tic_tac_toe_9_features-digits.csv"
			df.to_csv(file_path, index=False)
	print(np.unique(tic_tac_datas[:, 0], return_counts=True))
	return tic_tac_datas


def generate_tic_tac_toe_data2(n_samples=100,
							   all_features_flag=True,
							   save_flag=False,
							   use_all_combinations=False):
	"""
	n_samples: 当 use_all_combinations=False 时，才有意义，表示随机生成多少条数据
	all_features_flag: True则生成9个格点的特征，否则只生成5个格点
	save_flag: 是否保存为 CSV
	use_all_combinations: 是否穷举3^5或3^9的所有组合
	"""
	import itertools
	# 这里统一设定 X=0.3, O=0.6, Empty=0.9
	# 如果你的实际需求里有不同数值映射，可以在这儿自行调整
	X_val = 0.3
	O_val = 0.6
	E_val = 0.9

	data = []

	# 如果只需要 5 个特征，对应的位置如下
	# [Top-Left, Top-Middle, Middle-Middle, Bottom-Left, Bottom-Right]
	# 如果需要 9 个特征，对应所有井字格如下
	# [Top-Left, Top-Middle, Top-Right,
	#  Middle-Left, Middle-Middle, Middle-Right,
	#  Bottom-Left, Bottom-Middle, Bottom-Right]

	if not all_features_flag:
		# ----------------
		# 只看 5 个格点
		# ----------------

		# 1) 生成所有可能的组合
		if use_all_combinations:
			# itertools.product 的 repeat=5，产生3^5=243种组合
			all_boards = itertools.product([X_val, O_val, E_val], repeat=5)
		else:
			# 仍然沿用随机逻辑
			import random
			all_boards = (
				(
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val])
				)
				for _ in range(n_samples)
			)

		for board in all_boards:
			top_left, top_middle, middle_middle, bottom_left, bottom_right = board

			# 简单规则判断胜负
			win_x = 0
			win_y = 0
			# 判断 X 胜
			if ((top_left == middle_middle == bottom_right == X_val) or
					(top_middle == middle_middle == bottom_left == X_val) or
					(middle_middle == bottom_left == bottom_right == X_val)):
				win_x = 1

			# 判断 O 胜
			if ((top_left == middle_middle == bottom_right == O_val) or
					(top_middle == middle_middle == bottom_left == O_val) or
					(middle_middle == bottom_left == bottom_right == O_val)):
				win_y = 1

			# 汇总结果
			if win_x and win_y:
				# 其实在5个格点的简化场景中，不太会出现 X 与 O 同时成线
				# 这里保留你的原逻辑
				win = 3
			elif win_x:
				win = 1
			elif win_y:
				win = 2
			else:
				win = 0
				continue  # 调试用
			data.append([
				win,
				top_left, top_middle,
				middle_middle,
				bottom_left, bottom_right
			])

		# 转成 numpy 数组
		tic_tac_datas = np.array(data, dtype=float)

		# 如果要保存
		if save_flag:
			df = pd.DataFrame(
				data,
				columns=[
					'Win',
					'Top-Left', 'Top-Middle',
					'Middle-Middle',
					'Bottom-Left', 'Bottom-Right'
				]
			)
			file_path = "tic_tac_toe_5_features-digits.csv"
			df.to_csv(file_path, index=False)

	else:
		# ----------------
		# 全 9 个格点
		# ----------------

		if use_all_combinations:
			# 3^9 = 19683 种组合
			all_boards = itertools.product([X_val, O_val, E_val], repeat=9)
		else:
			# 随机
			import random
			all_boards = (
				(
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val]),
					random.choice([X_val, O_val, E_val])
				)
				for _ in range(n_samples)
			)

		for board in all_boards:
			(top_left, top_middle, top_right,
			 middle_left, middle_middle, middle_right,
			 bottom_left, bottom_middle, bottom_right) = board

			win_x = 0
			win_y = 0

			# ------ 判断 X 是否胜利 ------
			lines_for_x = [
				# 行
				(top_left, top_middle, top_right),
				(middle_left, middle_middle, middle_right),
				(bottom_left, bottom_middle, bottom_right),
				# 列
				(top_left, middle_left, bottom_left),
				(top_middle, middle_middle, bottom_middle),
				(top_right, middle_right, bottom_right),
				# 2条对角线
				(top_left, middle_middle, bottom_right),
				(top_right, middle_middle, bottom_left),
			]
			if any(line == (X_val, X_val, X_val) for line in lines_for_x):
				win_x = 1

			# ------ 判断 O 是否胜利 ------
			lines_for_o = [
				# 行
				(top_left, top_middle, top_right),
				(middle_left, middle_middle, middle_right),
				(bottom_left, bottom_middle, bottom_right),
				# 列
				(top_left, middle_left, bottom_left),
				(top_middle, middle_middle, bottom_middle),
				(top_right, middle_right, bottom_right),
				# 2条对角线
				(top_left, middle_middle, bottom_right),
				(top_right, middle_middle, bottom_left),
			]
			if any(line == (O_val, O_val, O_val) for line in lines_for_o):
				win_y = 1

			# 最终 win 结果
			if win_x and win_y:
				# 同时出现 X, O 都成三连，设为 3
				# win = 3
				win = 1
			elif win_x:
				win = 1
			elif win_y:
				win = 2
			else:
				win = 0
				continue  # 调试用
			data.append([
				win,
				top_left, top_middle, top_right,
				middle_left, middle_middle, middle_right,
				bottom_left, bottom_middle, bottom_right
			])

		# 转成 numpy 数组
		tic_tac_datas = np.array(data, dtype=float)

		# 如果要保存
		if save_flag:
			df = pd.DataFrame(
				data,
				columns=[
					'Win',
					'Top-Left', 'Top-Middle', 'Top-Right',
					'Middle-Left', 'Middle-Middle', 'Middle-Right',
					'Bottom-Left', 'Bottom-Middle', 'Bottom-Right'
				]
			)
			file_path = "tic_tac_toe_9_features-digits.csv"
			df.to_csv(file_path, index=False)

	# 打印一下每种胜负值的分布情况
	print("Win 列中不同值的统计分布：", np.unique(tic_tac_datas[:, 0], return_counts=True))

	return tic_tac_datas


def repeat_rows_numpy(data, numbers):
	"""
	通过循环填充数据，使数据行数达到 numbers
	:param data: 原始 NumPy 数组
	:param numbers: 目标行数
	:return: 扩展后的 NumPy 数组
	"""
	num_rows = data.shape[0]  # 原始数据行数
	repeat_times = (numbers // num_rows) + 1  # 计算需要重复的次数
	expanded_data = np.tile(data, (repeat_times, 1))  # 循环扩展数据
	return expanded_data[:numbers]  # 取前 numbers 行


def generate_data_titanic(display_flag=False, save_flag=False):
	import pandas as pd
	import numpy as np
	# 下载数据集
	url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
	data = pd.read_csv(url)

	# 删除不需要的特征
	data = data.drop(["Name", "Fare"], axis=1)

	# 填充数值特征的缺失值
	data["Age"].fillna(data["Age"].median(), inplace=True)

	# data["Fare"].fillna(data["Fare"].median(), inplace=True)

	# 离散化 Fare （票价）

	# 离散化 Age（年龄）
	def age_category(age):
		if age < 12:
			return "Child"
		elif age < 18:
			return "Teen"
		elif age < 60:
			return "Adult"
		else:
			return "Senior"

	data["Age_Category"] = data["Age"].apply(age_category)
	data = data.drop(["Age"], axis=1)

	def categorize_siblings_spouses(value):
		"""
		将 Siblings/Spouses Aboard 特征离散化：
		- 0  -> "None"
		- 1  -> "Only 1"
		- 2+ -> "More than 1"
		"""
		if value == 0:
			return "None"
		elif value == 1:
			return "Only 1"
		else:
			return "More than 1"

	def categorize_parents_children(value):
		"""
		将 Siblings/Spouses Aboard 特征离散化：
		- 0  -> "None"
		- 1  -> "Only 1"
		- 2+ -> "More than 1"
		"""
		if value == 0:
			return "None"
		elif value == 1:
			return "Only 1"
		else:
			return "More than 1"

	# 示例数据
	# data = pd.DataFrame({'Siblings/Spouses Aboard': [0, 1, 2, 3, 4, 5, 8]})

	# 应用离散化
	data['Siblings_Spouses_Category'] = data['Siblings/Spouses Aboard'].apply(categorize_siblings_spouses)
	data = data.drop(["Siblings/Spouses Aboard"], axis=1)

	data['Parents/Children Aboard_Category'] = data['Parents/Children Aboard'].apply(categorize_parents_children)
	data = data.drop(["Parents/Children Aboard"], axis=1)

	def encode_categories(data):
		"""
		将分类数据转换为数值：
		- Sex: 'female' -> 0, 'male' -> 1
		- Age_Category: 'Adult' -> 0, 'Child' -> 1, 'Senior' -> 2, 'Teen' -> 3
		- Siblings_Spouses_Category: 'More than 1' -> 0, 'None' -> 1, 'Only 1' -> 2
		- Parents/Children Aboard_Category: 'More than 1' -> 0, 'None' -> 1, 'Only 1' -> 2
		"""

		# 定义映射字典
		mappings = {
			"Sex": {"female": 0, "male": 1},
			"Age_Category": {"Adult": 0, "Child": 1, "Senior": 2, "Teen": 3},
			"Siblings_Spouses_Category": {"More than 1": 0, "None": 1, "Only 1": 2},
			"Parents/Children Aboard_Category": {"More than 1": 0, "None": 1, "Only 1": 2},
		}

		# 遍历字典进行映射
		for col, mapping in mappings.items():
			if col in data.columns:
				data[col] = data[col].map(mapping)

		return data

	data = encode_categories(data)
	if display_flag:
		# 显示结果
		print(data)
		for i in range(data.shape[1]):
			print(data.iloc[:, i].value_counts())
	# Survived
	# 0    545
	# 1    342
	# Name: count, dtype: int64
	# Pclass
	# 3    487
	# 1    216
	# 2    184
	# Name: count, dtype: int64
	# Sex
	# 1    573
	# 0    314
	# Name: count, dtype: int64
	# Age_Category
	# 0    726
	# 1     77
	# 3     53
	# 2     31
	# Name: count, dtype: int64
	# Siblings_Spouses_Category
	# 1    604
	# 2    209
	# 0     74
	# Name: count, dtype: int64
	# Parents/Children Aboard_Category
	# 1    674
	# 2    118
	# 0     95
	# Name: count, dtype: int64

	if save_flag:
		file_path = "./titanic.csv"
		data.to_csv(file_path, index=False)
	return np.array(data)


def normalize_columns(data):
	"""
	对 NumPy 数组的每列进行归一化：
	- 计算每列的最大值
	- 进行归一化 (value / max_value)
	- 处理可能的 0 值，防止除 0 错误

	参数：
	- data: NumPy 数组 (2D)

	返回：
	- normalized_data: 归一化后的 NumPy 数组
	- max_values: 每列的最大值（用于恢复原始数据）
	"""
	max_values = np.max(data, axis=0)  # 计算每列最大值
	max_values[max_values == 0] = 1  # 防止除 0 错误
	normalized_data = data / max_values  # 归一化处理
	return normalized_data, max_values
