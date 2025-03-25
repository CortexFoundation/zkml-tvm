""" MRT operator names """

VAR = "var"

DROP_OUT = "nn.dropout"
CONV2D = "nn.conv2d"
DENSE = "nn.dense"
BATCH_NORM = "nn.batch_norm"
BIAS_ADD = "nn.bias_add"
RELU = "nn.relu"
LEAKY_RELU = "nn.leaky_relu"
ADAPTIVE_AVG_POOL2D = "nn.adaptive_avg_pool2d"
AVG_POOL2D = "nn.avg_pool2d"
MAX_POOL2D = "nn.max_pool2d"

SOFTMAX = "nn.softmax"
LOG_SOFTMAX = "nn.log_softmax"

EXP = "exp"
SIGMOID = "sigmoid"

SUM = "sum"
MEAN = "mean"
MAXIMUM = "maximum"
MINIMUM = "minimum"

# =========== NON-CALC ops ===============
TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"

REPEAT = "repeat"
SQUEEZE = "squeeze"
FLATTEN = "flatten"
BATCH_FLATTEN = "nn.batch_flatten"
RESHAPE = "reshape"
CONCAT = "concatenate"
SPLIT = "split"
TRANSPOSE = "transpose"

EXPAND_DIMS = "expand_dims"
TILE = "tile"

WHERE = "where"
GREATER = "greater"
STRIDED_SLICE = "strided_slice"
SLICE_LIKE = "slice_like"
GET_VALID_COUNT = "vision.get_valid_counts"
NON_MAX_SUPRESSION = "vision.non_max_suppression"

CLIP = "clip"
CEIL = "ceil"
RIGHT_SHIFT = "right_shift"
# AS_TYPE = "astype"
CAST = "cast"

ADV_INDEX = "adv_index"

# ======= binary ops =============

ADD = "add"
SUB = "subtract"
MUL = "multiply"
DIV = "divide"

# ======= unary ops ==============

NEGATIVE = "negative"

# ======= auto generate op =========
ARANGE = "arange"
ZEROS_LIKE = "zeros_like"
ONES_LIKE = "ones_like"

# ======= control flow op ===========
IF = "if"
ARGWHERE = "argwhere"

# ======= mrt requant op ==========
REQUANT = "mrt.requant"
PCLIP = "mrt.pclip"
""" precision clip """
RS_PCLIP = "mrt.rs_pclip"
""" right shift precision clip """
LUT = "mrt.lut"
""" look up table, equals adv_index in tvm """

