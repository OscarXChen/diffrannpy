# 真值表定义规范如下：
# A1 A0 B1 B0 L1 L2 L3 X1 X0
# A1为1，则输入A=1，A0为1，则输入A=0
# X1为1，则输出X为1，X0为1，则输出X为0
# 显然，这套规则下，A1·A0=0，不可能同时为1
def get_true_table_basics():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1, ]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_basics_2():
	# 论文4.1
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_func_complex_1():
	# 论文4.2
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 1,
		0, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0,
	]).reshape(-1, 6)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_func_adder():
	# 论文4.3
	import numpy as np
	inputs = np.array([
		# 0, 1, 0, 1, 0, 1,
		# 0, 1, 0, 1, 1, 0,
		# 0, 1, 1, 0, 0, 1,
		# 0, 1, 1, 0, 1, 0,
		# 1, 0, 0, 1, 0, 1,
		# 1, 0, 0, 1, 1, 0,
		# 1, 0, 1, 0, 0, 1,
		# 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1,
		0, 1, 1, 0, 1, 0,
		1, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 1,
		1, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 1, 0,

	]
	).reshape(-1, 6)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 1, 0, 0,
		1, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,

	]).reshape(-1, 4)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_logic_func_adder_expand():
	# 论文4.3
	import numpy as np
	inputs = np.array([
		# 0, 1, 0, 1, 0, 1,
		# 0, 1, 0, 1, 1, 0,
		# 0, 1, 1, 0, 0, 1,
		# 0, 1, 1, 0, 1, 0,
		# 1, 0, 0, 1, 0, 1,
		# 1, 0, 0, 1, 1, 0,
		# 1, 0, 1, 0, 0, 1,
		# 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1, 1,
		0, 1, 0, 1, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 0,
		0, 1, 1, 0, 1, 0, 0,
		1, 0, 0, 1, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 0,
		1, 0, 1, 0, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 0, 0,
		0, 1, 1, 0, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 0,
		1, 0, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 0,

	]
	).reshape(-1, 7)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 1, 0, 0,
		1, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,

	]).reshape(-1, 4)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_NAND_NOR_XOR():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1, ]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_xor_nxor_nand():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		1, 0,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor2():
	import numpy as np
	inputs = np.array([
		1, 0, 1,
		1, 0, 0,
		0, 1, 1,
		0, 1, 0,
	]
	).reshape(-1, 3)
	logic_control = np.array([
		0,
		1,
		0,
		1,
	]).reshape(-1, 1)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor3():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0, 1,
		1, 0, 1, 0, 0,
		1, 0, 0, 1, 1,
		1, 0, 0, 1, 0,
		0, 1, 1, 0, 1,
		0, 1, 1, 0, 0,
		0, 1, 0, 1, 1,
		0, 1, 0, 1, 0,
	]
	).reshape(-1, 5)
	logic_control = np.array([
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1
	]).reshape(-1, 1)
	outputs = np.array([
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor3_kernel():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 1,
		0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 0, 1, 0, 1,
	]
	).reshape(-1, 6)
	logic_control = np.array([
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
	]).reshape(-1, 1)
	outputs = np.array([
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_shift_test1():
	import numpy as np
	inputs = np.array([
		1,
	]
	).reshape(-1, 1)
	logic_control = np.array([
		0,

	]).reshape(-1, 1)
	outputs = np.array([
		0, 1

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_function():
	# 暂时不用
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_test1():
	# 暂时不用
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 1, 1, 1,

	]
	).reshape(-1, 4)
	# logic_control = np.array([
	# 	1, 0, 0,
	# 	0, 1, 0,
	# 	0, 0, 1,
	# ]).reshape(-1, 3)
	logic_control = np.array([

	]).reshape(-1, 3)
	outputs = np.array([
		0, 1, 0,
		1, 0, 0,
		0, 0, 1,
	]).reshape(-1, 3)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs
