»ë
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ÍÑ

HiddenLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*$
shared_nameHiddenLayer1/kernel
}
'HiddenLayer1/kernel/Read/ReadVariableOpReadVariableOpHiddenLayer1/kernel* 
_output_shapes
:
¬*
dtype0
{
HiddenLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*"
shared_nameHiddenLayer1/bias
t
%HiddenLayer1/bias/Read/ReadVariableOpReadVariableOpHiddenLayer1/bias*
_output_shapes	
:¬*
dtype0

HiddenLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*$
shared_nameHiddenLayer2/kernel
}
'HiddenLayer2/kernel/Read/ReadVariableOpReadVariableOpHiddenLayer2/kernel* 
_output_shapes
:
¬*
dtype0
{
HiddenLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameHiddenLayer2/bias
t
%HiddenLayer2/bias/Read/ReadVariableOpReadVariableOpHiddenLayer2/bias*
_output_shapes	
:*
dtype0

OutputLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*#
shared_nameOutputLayer/kernel
z
&OutputLayer/kernel/Read/ReadVariableOpReadVariableOpOutputLayer/kernel*
_output_shapes
:	
*
dtype0
x
OutputLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameOutputLayer/bias
q
$OutputLayer/bias/Read/ReadVariableOpReadVariableOpOutputLayer/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
­
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*è
valueÞBÛ BÔ
ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
6
!iter
	"decay
#learning_rate
$momentum
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
­

%layers
&metrics
'non_trainable_variables
trainable_variables
(layer_metrics
)layer_regularization_losses
	variables
regularization_losses
 
 
 
 
­

*layers
+metrics
,non_trainable_variables
trainable_variables
-layer_metrics
.layer_regularization_losses
	variables
regularization_losses
_]
VARIABLE_VALUEHiddenLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

/layers
0metrics
1non_trainable_variables
trainable_variables
2layer_metrics
3layer_regularization_losses
	variables
regularization_losses
_]
VARIABLE_VALUEHiddenLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEHiddenLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

4layers
5metrics
6non_trainable_variables
trainable_variables
7layer_metrics
8layer_regularization_losses
	variables
regularization_losses
^\
VARIABLE_VALUEOutputLayer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEOutputLayer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­

9layers
:metrics
;non_trainable_variables
trainable_variables
<layer_metrics
=layer_regularization_losses
	variables
regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

>0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	@total
	Acount
B	variables
C	keras_api
D
	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

B	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

G	variables

 serving_default_InputLayer_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
¿
StatefulPartitionedCallStatefulPartitionedCall serving_default_InputLayer_inputHiddenLayer1/kernelHiddenLayer1/biasHiddenLayer2/kernelHiddenLayer2/biasOutputLayer/kernelOutputLayer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_92444
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'HiddenLayer1/kernel/Read/ReadVariableOp%HiddenLayer1/bias/Read/ReadVariableOp'HiddenLayer2/kernel/Read/ReadVariableOp%HiddenLayer2/bias/Read/ReadVariableOp&OutputLayer/kernel/Read/ReadVariableOp$OutputLayer/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_92668
ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHiddenLayer1/kernelHiddenLayer1/biasHiddenLayer2/kernelHiddenLayer2/biasOutputLayer/kernelOutputLayer/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_92720½
µ
a
E__inference_InputLayer_layer_call_and_return_conditional_losses_92538

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
»
*__inference_sequential_layer_call_fn_92515

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_923672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú&
é
__inference__traced_save_92668
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fef64cdae34d4e2097cca1fb70fc8bc8/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¦
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*^
_input_shapesM
K: :
¬:¬:
¬::	
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ü
¾
#__inference_signature_wrapper_92444
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_922242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input
¶
®
F__inference_OutputLayer_layer_call_and_return_conditional_losses_92307

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â<

!__inference__traced_restore_92720
file_prefix(
$assignvariableop_hiddenlayer1_kernel(
$assignvariableop_1_hiddenlayer1_bias*
&assignvariableop_2_hiddenlayer2_kernel(
$assignvariableop_3_hiddenlayer2_bias)
%assignvariableop_4_outputlayer_kernel'
#assignvariableop_5_outputlayer_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
identity_15¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity£
AssignVariableOpAssignVariableOp$assignvariableop_hiddenlayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1©
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2«
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hiddenlayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3©
AssignVariableOp_3AssignVariableOp$assignvariableop_3_hiddenlayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_outputlayer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_outputlayer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6 
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¡
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
µ
¯
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_92574

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
É
æ
E__inference_sequential_layer_call_and_return_conditional_losses_92404

inputs
hiddenlayer1_92388
hiddenlayer1_92390
hiddenlayer2_92393
hiddenlayer2_92395
outputlayer_92398
outputlayer_92400
identity¢$HiddenLayer1/StatefulPartitionedCall¢$HiddenLayer2/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCallÝ
InputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_InputLayer_layer_call_and_return_conditional_losses_922342
InputLayer/PartitionedCallÆ
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#InputLayer/PartitionedCall:output:0hiddenlayer1_92388hiddenlayer1_92390*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_922532&
$HiddenLayer1/StatefulPartitionedCallÐ
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_92393hiddenlayer2_92395*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_922802&
$HiddenLayer2/StatefulPartitionedCallÊ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0outputlayer_92398outputlayer_92400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_OutputLayer_layer_call_and_return_conditional_losses_923072%
#OutputLayer/StatefulPartitionedCallô
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

,__inference_HiddenLayer1_layer_call_fn_92563

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_922532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
*__inference_sequential_layer_call_fn_92382
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_923672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input
¡
F
*__inference_InputLayer_layer_call_fn_92543

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_InputLayer_layer_call_and_return_conditional_losses_922342
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

,__inference_HiddenLayer2_layer_call_fn_92583

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_922802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
µ
¯
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_92280

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
µ
¯
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_92253

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
æ
E__inference_sequential_layer_call_and_return_conditional_losses_92367

inputs
hiddenlayer1_92351
hiddenlayer1_92353
hiddenlayer2_92356
hiddenlayer2_92358
outputlayer_92361
outputlayer_92363
identity¢$HiddenLayer1/StatefulPartitionedCall¢$HiddenLayer2/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCallÝ
InputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_InputLayer_layer_call_and_return_conditional_losses_922342
InputLayer/PartitionedCallÆ
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#InputLayer/PartitionedCall:output:0hiddenlayer1_92351hiddenlayer1_92353*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_922532&
$HiddenLayer1/StatefulPartitionedCallÐ
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_92356hiddenlayer2_92358*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_922802&
$HiddenLayer2/StatefulPartitionedCallÊ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0outputlayer_92361outputlayer_92363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_OutputLayer_layer_call_and_return_conditional_losses_923072%
#OutputLayer/StatefulPartitionedCallô
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

E__inference_sequential_layer_call_and_return_conditional_losses_92471

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identityu
InputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
InputLayer/Const
InputLayer/ReshapeReshapeinputsInputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
InputLayer/Reshape¶
"HiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"HiddenLayer1/MatMul/ReadVariableOp°
HiddenLayer1/MatMulMatMulInputLayer/Reshape:output:0*HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/MatMul´
#HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#HiddenLayer1/BiasAdd/ReadVariableOp¶
HiddenLayer1/BiasAddBiasAddHiddenLayer1/MatMul:product:0+HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/BiasAdd
HiddenLayer1/ReluReluHiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/Relu¶
"HiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"HiddenLayer2/MatMul/ReadVariableOp´
HiddenLayer2/MatMulMatMulHiddenLayer1/Relu:activations:0*HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/MatMul´
#HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#HiddenLayer2/BiasAdd/ReadVariableOp¶
HiddenLayer2/BiasAddBiasAddHiddenLayer2/MatMul:product:0+HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/BiasAdd
HiddenLayer2/ReluReluHiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/Relu²
!OutputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02#
!OutputLayer/MatMul/ReadVariableOp°
OutputLayer/MatMulMatMulHiddenLayer2/Relu:activations:0)OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/MatMul°
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"OutputLayer/BiasAdd/ReadVariableOp±
OutputLayer/BiasAddBiasAddOutputLayer/MatMul:product:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/BiasAdd
OutputLayer/SoftmaxSoftmaxOutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/Softmaxq
IdentityIdentityOutputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ:::::::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
!
²
 __inference__wrapped_model_92224
inputlayer_input:
6sequential_hiddenlayer1_matmul_readvariableop_resource;
7sequential_hiddenlayer1_biasadd_readvariableop_resource:
6sequential_hiddenlayer2_matmul_readvariableop_resource;
7sequential_hiddenlayer2_biasadd_readvariableop_resource9
5sequential_outputlayer_matmul_readvariableop_resource:
6sequential_outputlayer_biasadd_readvariableop_resource
identity
sequential/InputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential/InputLayer/Const´
sequential/InputLayer/ReshapeReshapeinputlayer_input$sequential/InputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/InputLayer/Reshape×
-sequential/HiddenLayer1/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02/
-sequential/HiddenLayer1/MatMul/ReadVariableOpÜ
sequential/HiddenLayer1/MatMulMatMul&sequential/InputLayer/Reshape:output:05sequential/HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
sequential/HiddenLayer1/MatMulÕ
.sequential/HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype020
.sequential/HiddenLayer1/BiasAdd/ReadVariableOpâ
sequential/HiddenLayer1/BiasAddBiasAdd(sequential/HiddenLayer1/MatMul:product:06sequential/HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
sequential/HiddenLayer1/BiasAdd¡
sequential/HiddenLayer1/ReluRelu(sequential/HiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential/HiddenLayer1/Relu×
-sequential/HiddenLayer2/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02/
-sequential/HiddenLayer2/MatMul/ReadVariableOpà
sequential/HiddenLayer2/MatMulMatMul*sequential/HiddenLayer1/Relu:activations:05sequential/HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential/HiddenLayer2/MatMulÕ
.sequential/HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential/HiddenLayer2/BiasAdd/ReadVariableOpâ
sequential/HiddenLayer2/BiasAddBiasAdd(sequential/HiddenLayer2/MatMul:product:06sequential/HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/HiddenLayer2/BiasAdd¡
sequential/HiddenLayer2/ReluRelu(sequential/HiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/HiddenLayer2/ReluÓ
,sequential/OutputLayer/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02.
,sequential/OutputLayer/MatMul/ReadVariableOpÜ
sequential/OutputLayer/MatMulMatMul*sequential/HiddenLayer2/Relu:activations:04sequential/OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential/OutputLayer/MatMulÑ
-sequential/OutputLayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential/OutputLayer/BiasAdd/ReadVariableOpÝ
sequential/OutputLayer/BiasAddBiasAdd'sequential/OutputLayer/MatMul:product:05sequential/OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential/OutputLayer/BiasAdd¦
sequential/OutputLayer/SoftmaxSoftmax'sequential/OutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential/OutputLayer/Softmax|
IdentityIdentity(sequential/OutputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ:::::::] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input

Å
*__inference_sequential_layer_call_fn_92419
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_924042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input
æ

+__inference_OutputLayer_layer_call_fn_92603

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_OutputLayer_layer_call_and_return_conditional_losses_923072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
¯
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_92554

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ð
E__inference_sequential_layer_call_and_return_conditional_losses_92344
inputlayer_input
hiddenlayer1_92328
hiddenlayer1_92330
hiddenlayer2_92333
hiddenlayer2_92335
outputlayer_92338
outputlayer_92340
identity¢$HiddenLayer1/StatefulPartitionedCall¢$HiddenLayer2/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCallç
InputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_InputLayer_layer_call_and_return_conditional_losses_922342
InputLayer/PartitionedCallÆ
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#InputLayer/PartitionedCall:output:0hiddenlayer1_92328hiddenlayer1_92330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_922532&
$HiddenLayer1/StatefulPartitionedCallÐ
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_92333hiddenlayer2_92335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_922802&
$HiddenLayer2/StatefulPartitionedCallÊ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0outputlayer_92338outputlayer_92340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_OutputLayer_layer_call_and_return_conditional_losses_923072%
#OutputLayer/StatefulPartitionedCallô
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input
µ
a
E__inference_InputLayer_layer_call_and_return_conditional_losses_92234

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
»
*__inference_sequential_layer_call_fn_92532

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_924042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ð
E__inference_sequential_layer_call_and_return_conditional_losses_92324
inputlayer_input
hiddenlayer1_92264
hiddenlayer1_92266
hiddenlayer2_92291
hiddenlayer2_92293
outputlayer_92318
outputlayer_92320
identity¢$HiddenLayer1/StatefulPartitionedCall¢$HiddenLayer2/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCallç
InputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_InputLayer_layer_call_and_return_conditional_losses_922342
InputLayer/PartitionedCallÆ
$HiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#InputLayer/PartitionedCall:output:0hiddenlayer1_92264hiddenlayer1_92266*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_922532&
$HiddenLayer1/StatefulPartitionedCallÐ
$HiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer1/StatefulPartitionedCall:output:0hiddenlayer2_92291hiddenlayer2_92293*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_922802&
$HiddenLayer2/StatefulPartitionedCallÊ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall-HiddenLayer2/StatefulPartitionedCall:output:0outputlayer_92318outputlayer_92320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_OutputLayer_layer_call_and_return_conditional_losses_923072%
#OutputLayer/StatefulPartitionedCallô
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0%^HiddenLayer1/StatefulPartitionedCall%^HiddenLayer2/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ::::::2L
$HiddenLayer1/StatefulPartitionedCall$HiddenLayer1/StatefulPartitionedCall2L
$HiddenLayer2/StatefulPartitionedCall$HiddenLayer2/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInputLayer_input
¶
®
F__inference_OutputLayer_layer_call_and_return_conditional_losses_92594

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

E__inference_sequential_layer_call_and_return_conditional_losses_92498

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identityu
InputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
InputLayer/Const
InputLayer/ReshapeReshapeinputsInputLayer/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
InputLayer/Reshape¶
"HiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"HiddenLayer1/MatMul/ReadVariableOp°
HiddenLayer1/MatMulMatMulInputLayer/Reshape:output:0*HiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/MatMul´
#HiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#HiddenLayer1/BiasAdd/ReadVariableOp¶
HiddenLayer1/BiasAddBiasAddHiddenLayer1/MatMul:product:0+HiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/BiasAdd
HiddenLayer1/ReluReluHiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
HiddenLayer1/Relu¶
"HiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"HiddenLayer2/MatMul/ReadVariableOp´
HiddenLayer2/MatMulMatMulHiddenLayer1/Relu:activations:0*HiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/MatMul´
#HiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#HiddenLayer2/BiasAdd/ReadVariableOp¶
HiddenLayer2/BiasAddBiasAddHiddenLayer2/MatMul:product:0+HiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/BiasAdd
HiddenLayer2/ReluReluHiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
HiddenLayer2/Relu²
!OutputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02#
!OutputLayer/MatMul/ReadVariableOp°
OutputLayer/MatMulMatMulHiddenLayer2/Relu:activations:0)OutputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/MatMul°
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"OutputLayer/BiasAdd/ReadVariableOp±
OutputLayer/BiasAddBiasAddOutputLayer/MatMul:product:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/BiasAdd
OutputLayer/SoftmaxSoftmaxOutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
OutputLayer/Softmaxq
IdentityIdentityOutputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ:::::::S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
Q
InputLayer_input=
"serving_default_InputLayer_input:0ÿÿÿÿÿÿÿÿÿ?
OutputLayer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:
¿$
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
I__call__
J_default_save_signature
*K&call_and_return_all_conditional_losses"ò!
_tf_keras_sequentialÓ!{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "InputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "InputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "OutputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "InputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "InputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "OutputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
ã
trainable_variables
	variables
regularization_losses
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Flatten", "name": "InputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "InputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ý

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dense", "name": "HiddenLayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
ý

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dense", "name": "HiddenLayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 150, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
ý

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
R__call__
*S&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dense", "name": "OutputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "OutputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
I
!iter
	"decay
#learning_rate
$momentum"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê

%layers
&metrics
'non_trainable_variables
trainable_variables
(layer_metrics
)layer_regularization_losses
	variables
regularization_losses
I__call__
J_default_save_signature
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
,
Tserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

*layers
+metrics
,non_trainable_variables
trainable_variables
-layer_metrics
.layer_regularization_losses
	variables
regularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
':%
¬2HiddenLayer1/kernel
 :¬2HiddenLayer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

/layers
0metrics
1non_trainable_variables
trainable_variables
2layer_metrics
3layer_regularization_losses
	variables
regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
':%
¬2HiddenLayer2/kernel
 :2HiddenLayer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

4layers
5metrics
6non_trainable_variables
trainable_variables
7layer_metrics
8layer_regularization_losses
	variables
regularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
%:#	
2OutputLayer/kernel
:
2OutputLayer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

9layers
:metrics
;non_trainable_variables
trainable_variables
<layer_metrics
=layer_regularization_losses
	variables
regularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
<
0
1
2
3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
»
	@total
	Acount
B	variables
C	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	Dtotal
	Ecount
F
_fn_kwargs
G	variables
H	keras_api"¿
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
@0
A1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
-
G	variables"
_generic_user_object
ö2ó
*__inference_sequential_layer_call_fn_92515
*__inference_sequential_layer_call_fn_92382
*__inference_sequential_layer_call_fn_92532
*__inference_sequential_layer_call_fn_92419À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
 __inference__wrapped_model_92224Ã
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_92498
E__inference_sequential_layer_call_and_return_conditional_losses_92344
E__inference_sequential_layer_call_and_return_conditional_losses_92471
E__inference_sequential_layer_call_and_return_conditional_losses_92324À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_InputLayer_layer_call_fn_92543¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_InputLayer_layer_call_and_return_conditional_losses_92538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_HiddenLayer1_layer_call_fn_92563¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_92554¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_HiddenLayer2_layer_call_fn_92583¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_92574¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_OutputLayer_layer_call_fn_92603¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_OutputLayer_layer_call_and_return_conditional_losses_92594¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
;B9
#__inference_signature_wrapper_92444InputLayer_input©
G__inference_HiddenLayer1_layer_call_and_return_conditional_losses_92554^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 
,__inference_HiddenLayer1_layer_call_fn_92563Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬©
G__inference_HiddenLayer2_layer_call_and_return_conditional_losses_92574^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_HiddenLayer2_layer_call_fn_92583Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_InputLayer_layer_call_and_return_conditional_losses_92538]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_InputLayer_layer_call_fn_92543P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_OutputLayer_layer_call_and_return_conditional_losses_92594]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
+__inference_OutputLayer_layer_call_fn_92603P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
§
 __inference__wrapped_model_92224=¢:
3¢0
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
OutputLayer%"
OutputLayerÿÿÿÿÿÿÿÿÿ
¿
E__inference_sequential_layer_call_and_return_conditional_losses_92324vE¢B
;¢8
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¿
E__inference_sequential_layer_call_and_return_conditional_losses_92344vE¢B
;¢8
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
E__inference_sequential_layer_call_and_return_conditional_losses_92471l;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
E__inference_sequential_layer_call_and_return_conditional_losses_92498l;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
*__inference_sequential_layer_call_fn_92382iE¢B
;¢8
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_92419iE¢B
;¢8
.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_92515_;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

*__inference_sequential_layer_call_fn_92532_;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¾
#__inference_signature_wrapper_92444Q¢N
¢ 
GªD
B
InputLayer_input.+
InputLayer_inputÿÿÿÿÿÿÿÿÿ"9ª6
4
OutputLayer%"
OutputLayerÿÿÿÿÿÿÿÿÿ
