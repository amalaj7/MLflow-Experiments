ΙΚ
Ρ£
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
Ύ
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878«

hiddenLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*$
shared_namehiddenLayer1/kernel
}
'hiddenLayer1/kernel/Read/ReadVariableOpReadVariableOphiddenLayer1/kernel* 
_output_shapes
:
¬*
dtype0
{
hiddenLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*"
shared_namehiddenLayer1/bias
t
%hiddenLayer1/bias/Read/ReadVariableOpReadVariableOphiddenLayer1/bias*
_output_shapes	
:¬*
dtype0

hiddenLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬d*$
shared_namehiddenLayer2/kernel
|
'hiddenLayer2/kernel/Read/ReadVariableOpReadVariableOphiddenLayer2/kernel*
_output_shapes
:	¬d*
dtype0
z
hiddenLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namehiddenLayer2/bias
s
%hiddenLayer2/bias/Read/ReadVariableOpReadVariableOphiddenLayer2/bias*
_output_shapes
:d*
dtype0

outputLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*#
shared_nameoutputLayer/kernel
y
&outputLayer/kernel/Read/ReadVariableOpReadVariableOpoutputLayer/kernel*
_output_shapes

:d
*
dtype0
x
outputLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameoutputLayer/bias
q
$outputLayer/bias/Read/ReadVariableOpReadVariableOpoutputLayer/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

Adam/hiddenLayer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*+
shared_nameAdam/hiddenLayer1/kernel/m

.Adam/hiddenLayer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/kernel/m* 
_output_shapes
:
¬*
dtype0

Adam/hiddenLayer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*)
shared_nameAdam/hiddenLayer1/bias/m

,Adam/hiddenLayer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/bias/m*
_output_shapes	
:¬*
dtype0

Adam/hiddenLayer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬d*+
shared_nameAdam/hiddenLayer2/kernel/m

.Adam/hiddenLayer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/kernel/m*
_output_shapes
:	¬d*
dtype0

Adam/hiddenLayer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hiddenLayer2/bias/m

,Adam/hiddenLayer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/bias/m*
_output_shapes
:d*
dtype0

Adam/outputLayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
**
shared_nameAdam/outputLayer/kernel/m

-Adam/outputLayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/outputLayer/kernel/m*
_output_shapes

:d
*
dtype0

Adam/outputLayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/outputLayer/bias/m

+Adam/outputLayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/outputLayer/bias/m*
_output_shapes
:
*
dtype0

Adam/hiddenLayer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*+
shared_nameAdam/hiddenLayer1/kernel/v

.Adam/hiddenLayer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/kernel/v* 
_output_shapes
:
¬*
dtype0

Adam/hiddenLayer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*)
shared_nameAdam/hiddenLayer1/bias/v

,Adam/hiddenLayer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/bias/v*
_output_shapes	
:¬*
dtype0

Adam/hiddenLayer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬d*+
shared_nameAdam/hiddenLayer2/kernel/v

.Adam/hiddenLayer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/kernel/v*
_output_shapes
:	¬d*
dtype0

Adam/hiddenLayer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hiddenLayer2/bias/v

,Adam/hiddenLayer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/bias/v*
_output_shapes
:d*
dtype0

Adam/outputLayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
**
shared_nameAdam/outputLayer/kernel/v

-Adam/outputLayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/outputLayer/kernel/v*
_output_shapes

:d
*
dtype0

Adam/outputLayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/outputLayer/bias/v

+Adam/outputLayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/outputLayer/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Έ*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*σ)
valueι)Bζ) Bί)

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
¬
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
 
­
+layer_metrics

,layers
-layer_regularization_losses
trainable_variables
.metrics
	variables
/non_trainable_variables
	regularization_losses
 
 
 
 
­
0layer_metrics
1layer_regularization_losses
trainable_variables
2metrics
	variables

3layers
4non_trainable_variables
regularization_losses
_]
VARIABLE_VALUEhiddenLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
5layer_metrics
6layer_regularization_losses
trainable_variables
7metrics
	variables

8layers
9non_trainable_variables
regularization_losses
 
 
 
­
:layer_metrics
;layer_regularization_losses
trainable_variables
<metrics
	variables

=layers
>non_trainable_variables
regularization_losses
_]
VARIABLE_VALUEhiddenLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
?layer_metrics
@layer_regularization_losses
trainable_variables
Ametrics
	variables

Blayers
Cnon_trainable_variables
regularization_losses
^\
VARIABLE_VALUEoutputLayer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEoutputLayer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
Dlayer_metrics
Elayer_regularization_losses
"trainable_variables
Fmetrics
#	variables

Glayers
Hnon_trainable_variables
$regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

I0
J1
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
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables

VARIABLE_VALUEAdam/hiddenLayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/hiddenLayer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/outputLayer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/outputLayer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/hiddenLayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/hiddenLayer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/outputLayer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/outputLayer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

 serving_default_inputLayer_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
Ώ
StatefulPartitionedCallStatefulPartitionedCall serving_default_inputLayer_inputhiddenLayer1/kernelhiddenLayer1/biashiddenLayer2/kernelhiddenLayer2/biasoutputLayer/kerneloutputLayer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_39117
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ο

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'hiddenLayer1/kernel/Read/ReadVariableOp%hiddenLayer1/bias/Read/ReadVariableOp'hiddenLayer2/kernel/Read/ReadVariableOp%hiddenLayer2/bias/Read/ReadVariableOp&outputLayer/kernel/Read/ReadVariableOp$outputLayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/hiddenLayer1/kernel/m/Read/ReadVariableOp,Adam/hiddenLayer1/bias/m/Read/ReadVariableOp.Adam/hiddenLayer2/kernel/m/Read/ReadVariableOp,Adam/hiddenLayer2/bias/m/Read/ReadVariableOp-Adam/outputLayer/kernel/m/Read/ReadVariableOp+Adam/outputLayer/bias/m/Read/ReadVariableOp.Adam/hiddenLayer1/kernel/v/Read/ReadVariableOp,Adam/hiddenLayer1/bias/v/Read/ReadVariableOp.Adam/hiddenLayer2/kernel/v/Read/ReadVariableOp,Adam/hiddenLayer2/bias/v/Read/ReadVariableOp-Adam/outputLayer/kernel/v/Read/ReadVariableOp+Adam/outputLayer/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
__inference__traced_save_39416
Ξ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehiddenLayer1/kernelhiddenLayer1/biashiddenLayer2/kernelhiddenLayer2/biasoutputLayer/kerneloutputLayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hiddenLayer1/kernel/mAdam/hiddenLayer1/bias/mAdam/hiddenLayer2/kernel/mAdam/hiddenLayer2/bias/mAdam/outputLayer/kernel/mAdam/outputLayer/bias/mAdam/hiddenLayer1/kernel/vAdam/hiddenLayer1/bias/vAdam/hiddenLayer2/kernel/vAdam/hiddenLayer2/bias/vAdam/outputLayer/kernel/vAdam/outputLayer/bias/v*'
Tin 
2*
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
!__inference__traced_restore_39507ο‘
α

E__inference_sequential_layer_call_and_return_conditional_losses_39037

inputs
hiddenlayer1_39020
hiddenlayer1_39022
hiddenlayer2_39026
hiddenlayer2_39028
outputlayer_39031
outputlayer_39033
identity’dropout/StatefulPartitionedCall’$hiddenLayer1/StatefulPartitionedCall’$hiddenLayer2/StatefulPartitionedCall’#outputLayer/StatefulPartitionedCallέ
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_388722
inputLayer/PartitionedCallΖ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39020hiddenlayer1_39022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_388912&
$hiddenLayer1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389192!
dropout/StatefulPartitionedCallΚ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hiddenlayer2_39026hiddenlayer2_39028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_389482&
$hiddenLayer2/StatefulPartitionedCallΚ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39031outputlayer_39033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_389752%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_39257

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????¬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????¬2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????¬2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????¬:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
 "
²
 __inference__wrapped_model_38862
inputlayer_input:
6sequential_hiddenlayer1_matmul_readvariableop_resource;
7sequential_hiddenlayer1_biasadd_readvariableop_resource:
6sequential_hiddenlayer2_matmul_readvariableop_resource;
7sequential_hiddenlayer2_biasadd_readvariableop_resource9
5sequential_outputlayer_matmul_readvariableop_resource:
6sequential_outputlayer_biasadd_readvariableop_resource
identity
sequential/inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential/inputLayer/Const΄
sequential/inputLayer/ReshapeReshapeinputlayer_input$sequential/inputLayer/Const:output:0*
T0*(
_output_shapes
:?????????2
sequential/inputLayer/ReshapeΧ
-sequential/hiddenLayer1/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02/
-sequential/hiddenLayer1/MatMul/ReadVariableOpά
sequential/hiddenLayer1/MatMulMatMul&sequential/inputLayer/Reshape:output:05sequential/hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2 
sequential/hiddenLayer1/MatMulΥ
.sequential/hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype020
.sequential/hiddenLayer1/BiasAdd/ReadVariableOpβ
sequential/hiddenLayer1/BiasAddBiasAdd(sequential/hiddenLayer1/MatMul:product:06sequential/hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2!
sequential/hiddenLayer1/BiasAdd‘
sequential/hiddenLayer1/ReluRelu(sequential/hiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????¬2
sequential/hiddenLayer1/Relu₯
sequential/dropout/IdentityIdentity*sequential/hiddenLayer1/Relu:activations:0*
T0*(
_output_shapes
:?????????¬2
sequential/dropout/IdentityΦ
-sequential/hiddenLayer2/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02/
-sequential/hiddenLayer2/MatMul/ReadVariableOpΩ
sequential/hiddenLayer2/MatMulMatMul$sequential/dropout/Identity:output:05sequential/hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential/hiddenLayer2/MatMulΤ
.sequential/hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential/hiddenLayer2/BiasAdd/ReadVariableOpα
sequential/hiddenLayer2/BiasAddBiasAdd(sequential/hiddenLayer2/MatMul:product:06sequential/hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential/hiddenLayer2/BiasAdd 
sequential/hiddenLayer2/ReluRelu(sequential/hiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential/hiddenLayer2/Relu?
,sequential/outputLayer/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02.
,sequential/outputLayer/MatMul/ReadVariableOpά
sequential/outputLayer/MatMulMatMul*sequential/hiddenLayer2/Relu:activations:04sequential/outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/outputLayer/MatMulΡ
-sequential/outputLayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential/outputLayer/BiasAdd/ReadVariableOpέ
sequential/outputLayer/BiasAddBiasAdd'sequential/outputLayer/MatMul:product:05sequential/outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential/outputLayer/BiasAdd¦
sequential/outputLayer/SoftmaxSoftmax'sequential/outputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
sequential/outputLayer/Softmax|
IdentityIdentity(sequential/outputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????:::::::] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
Ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_38924

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????¬2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????¬2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????¬:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
κ

,__inference_hiddenLayer1_layer_call_fn_39245

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_388912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Χ
π
E__inference_sequential_layer_call_and_return_conditional_losses_39013
inputlayer_input
hiddenlayer1_38996
hiddenlayer1_38998
hiddenlayer2_39002
hiddenlayer2_39004
outputlayer_39007
outputlayer_39009
identity’$hiddenLayer1/StatefulPartitionedCall’$hiddenLayer2/StatefulPartitionedCall’#outputLayer/StatefulPartitionedCallη
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_388722
inputLayer/PartitionedCallΖ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_38996hiddenlayer1_38998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_388912&
$hiddenLayer1/StatefulPartitionedCallϋ
dropout/PartitionedCallPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389242
dropout/PartitionedCallΒ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hiddenlayer2_39002hiddenlayer2_39004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_389482&
$hiddenLayer2/StatefulPartitionedCallΚ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39007outputlayer_39009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_389752%
#outputLayer/StatefulPartitionedCallτ
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
³
?
F__inference_outputLayer_layer_call_and_return_conditional_losses_39303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
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
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
΅
―
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39236

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
:?????????¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????¬2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Χ%

E__inference_sequential_layer_call_and_return_conditional_losses_39152

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identityu
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
inputLayer/Const
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:?????????2
inputLayer/ReshapeΆ
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp°
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/MatMul΄
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOpΆ
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/BiasAdd
hiddenLayer1/ReluReluhiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const₯
dropout/dropout/MulMulhiddenLayer1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/dropout/Mul}
dropout/dropout/ShapeShapehiddenLayer1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeΝ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????¬*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yί
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????¬2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????¬2
dropout/dropout/Mul_1΅
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp­
hiddenLayer2/MatMulMatMuldropout/dropout/Mul_1:z:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/MatMul³
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOp΅
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/BiasAdd
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/Relu±
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp°
outputLayer/MatMulMatMulhiddenLayer2/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/MatMul°
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp±
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/BiasAdd
outputLayer/SoftmaxSoftmaxoutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
outputLayer/Softmaxq
IdentityIdentityoutputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????:::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
θ

,__inference_hiddenLayer2_layer_call_fn_39292

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_389482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????¬::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_39272

inputs
identityΔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????¬:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs

Ε
*__inference_sequential_layer_call_fn_39052
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCallΆ
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_390372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
κ
»
*__inference_sequential_layer_call_fn_39214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_390752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
έ>
Λ
__inference__traced_save_39416
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_hiddenlayer1_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer1_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_m_read_readvariableop8
4savev2_adam_outputlayer_kernel_m_read_readvariableop6
2savev2_adam_outputlayer_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer1_bias_v_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_v_read_readvariableop8
4savev2_adam_outputlayer_kernel_v_read_readvariableop6
2savev2_adam_outputlayer_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
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
value3B1 B+_temp_7ff5b9ad9dd84397aa44f48d32c16379/part2	
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesΐ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesΙ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_hiddenlayer1_kernel_m_read_readvariableop3savev2_adam_hiddenlayer1_bias_m_read_readvariableop5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop3savev2_adam_hiddenlayer2_bias_m_read_readvariableop4savev2_adam_outputlayer_kernel_m_read_readvariableop2savev2_adam_outputlayer_bias_m_read_readvariableop5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop3savev2_adam_hiddenlayer1_bias_v_read_readvariableop5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop3savev2_adam_hiddenlayer2_bias_v_read_readvariableop4savev2_adam_outputlayer_kernel_v_read_readvariableop2savev2_adam_outputlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*Η
_input_shapes΅
²: :
¬:¬:	¬d:d:d
:
: : : : : : : : : :
¬:¬:	¬d:d:d
:
:
¬:¬:	¬d:d:d
:
: 2(
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
:¬:%!

_output_shapes
:	¬d: 

_output_shapes
:d:$ 

_output_shapes

:d
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
: :&"
 
_output_shapes
:
¬:!

_output_shapes	
:¬:%!

_output_shapes
:	¬d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:&"
 
_output_shapes
:
¬:!

_output_shapes	
:¬:%!

_output_shapes
:	¬d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: 
Ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_39262

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????¬2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????¬2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????¬:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
κ
»
*__inference_sequential_layer_call_fn_39197

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_390372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
΅
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_38872

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
χs
₯
!__inference__traced_restore_39507
file_prefix(
$assignvariableop_hiddenlayer1_kernel(
$assignvariableop_1_hiddenlayer1_bias*
&assignvariableop_2_hiddenlayer2_kernel(
$assignvariableop_3_hiddenlayer2_bias)
%assignvariableop_4_outputlayer_kernel'
#assignvariableop_5_outputlayer_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_12
.assignvariableop_15_adam_hiddenlayer1_kernel_m0
,assignvariableop_16_adam_hiddenlayer1_bias_m2
.assignvariableop_17_adam_hiddenlayer2_kernel_m0
,assignvariableop_18_adam_hiddenlayer2_bias_m1
-assignvariableop_19_adam_outputlayer_kernel_m/
+assignvariableop_20_adam_outputlayer_bias_m2
.assignvariableop_21_adam_hiddenlayer1_kernel_v0
,assignvariableop_22_adam_hiddenlayer1_bias_v2
.assignvariableop_23_adam_hiddenlayer2_kernel_v0
,assignvariableop_24_adam_hiddenlayer2_bias_v1
-assignvariableop_25_adam_outputlayer_kernel_v/
+assignvariableop_26_adam_outputlayer_bias_v
identity_28’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesΖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesΈ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
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

Identity_4ͺ
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

Identity_6‘
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9’
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11‘
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12‘
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ά
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_hiddenlayer1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16΄
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_hiddenlayer1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ά
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_hiddenlayer2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18΄
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_hiddenlayer2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19΅
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_outputlayer_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20³
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_outputlayer_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ά
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_hiddenlayer1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22΄
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_hiddenlayer1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ά
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_hiddenlayer2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24΄
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_hiddenlayer2_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25΅
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_outputlayer_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26³
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_outputlayer_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27£
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
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
―
―
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_38948

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????¬:::P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
ά
Ύ
#__inference_signature_wrapper_39117
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_388622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
‘
F
*__inference_inputLayer_layer_call_fn_39225

inputs
identityΗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_388722
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
δ

+__inference_outputLayer_layer_call_fn_39312

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_389752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_38919

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????¬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????¬2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????¬2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????¬:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
³
?
F__inference_outputLayer_layer_call_and_return_conditional_losses_38975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
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
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs

Ε
*__inference_sequential_layer_call_fn_39090
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCallΆ
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_390752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
΅
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_39220

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
―
―
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39283

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????¬:::P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
Ο

E__inference_sequential_layer_call_and_return_conditional_losses_39180

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identityu
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
inputLayer/Const
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:?????????2
inputLayer/ReshapeΆ
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp°
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/MatMul΄
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOpΆ
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/BiasAdd
hiddenLayer1/ReluReluhiddenLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????¬2
hiddenLayer1/Relu
dropout/IdentityIdentityhiddenLayer1/Relu:activations:0*
T0*(
_output_shapes
:?????????¬2
dropout/Identity΅
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource*
_output_shapes
:	¬d*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp­
hiddenLayer2/MatMulMatMuldropout/Identity:output:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/MatMul³
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOp΅
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/BiasAdd
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer2/Relu±
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp°
outputLayer/MatMulMatMulhiddenLayer2/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/MatMul°
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp±
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/BiasAdd
outputLayer/SoftmaxSoftmaxoutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
outputLayer/Softmaxq
IdentityIdentityoutputLayer/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????:::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
ζ
E__inference_sequential_layer_call_and_return_conditional_losses_39075

inputs
hiddenlayer1_39058
hiddenlayer1_39060
hiddenlayer2_39064
hiddenlayer2_39066
outputlayer_39069
outputlayer_39071
identity’$hiddenLayer1/StatefulPartitionedCall’$hiddenLayer2/StatefulPartitionedCall’#outputLayer/StatefulPartitionedCallέ
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_388722
inputLayer/PartitionedCallΖ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39058hiddenlayer1_39060*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_388912&
$hiddenLayer1/StatefulPartitionedCallϋ
dropout/PartitionedCallPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389242
dropout/PartitionedCallΒ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hiddenlayer2_39064hiddenlayer2_39066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_389482&
$hiddenLayer2/StatefulPartitionedCallΚ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_39069outputlayer_39071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_389752%
#outputLayer/StatefulPartitionedCallτ
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
΅
―
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_38891

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
:?????????¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¬2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????¬2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
‘
`
'__inference_dropout_layer_call_fn_39267

inputs
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????¬2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????¬22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????¬
 
_user_specified_nameinputs
?

E__inference_sequential_layer_call_and_return_conditional_losses_38992
inputlayer_input
hiddenlayer1_38902
hiddenlayer1_38904
hiddenlayer2_38959
hiddenlayer2_38961
outputlayer_38986
outputlayer_38988
identity’dropout/StatefulPartitionedCall’$hiddenLayer1/StatefulPartitionedCall’$hiddenLayer2/StatefulPartitionedCall’#outputLayer/StatefulPartitionedCallη
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_388722
inputLayer/PartitionedCallΖ
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_38902hiddenlayer1_38904*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_388912&
$hiddenLayer1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_389192!
dropout/StatefulPartitionedCallΚ
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hiddenlayer2_38959hiddenlayer2_38961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_389482&
$hiddenLayer2/StatefulPartitionedCallΚ
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0outputlayer_38986outputlayer_38988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_389752%
#outputLayer/StatefulPartitionedCall
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Δ
serving_default°
Q
inputLayer_input=
"serving_default_inputLayer_input:0??????????
outputLayer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:Μ«
­'
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
`__call__
*a&call_and_return_all_conditional_losses
b_default_save_signature"Σ$
_tf_keras_sequential΄${"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
γ
trainable_variables
	variables
regularization_losses
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"Τ
_tf_keras_layerΊ{"class_name": "Flatten", "name": "inputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ύ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Dense", "name": "hiddenLayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
α
trainable_variables
	variables
regularization_losses
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ύ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Dense", "name": "hiddenLayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
ύ

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
k__call__
*l&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Dense", "name": "outputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Ώ
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_"
	optimizer
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
+layer_metrics

,layers
-layer_regularization_losses
trainable_variables
.metrics
	variables
/non_trainable_variables
	regularization_losses
`__call__
b_default_save_signature
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
,
mserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
0layer_metrics
1layer_regularization_losses
trainable_variables
2metrics
	variables

3layers
4non_trainable_variables
regularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
':%
¬2hiddenLayer1/kernel
 :¬2hiddenLayer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5layer_metrics
6layer_regularization_losses
trainable_variables
7metrics
	variables

8layers
9non_trainable_variables
regularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:layer_metrics
;layer_regularization_losses
trainable_variables
<metrics
	variables

=layers
>non_trainable_variables
regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
&:$	¬d2hiddenLayer2/kernel
:d2hiddenLayer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?layer_metrics
@layer_regularization_losses
trainable_variables
Ametrics
	variables

Blayers
Cnon_trainable_variables
regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
$:"d
2outputLayer/kernel
:
2outputLayer/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dlayer_metrics
Elayer_regularization_losses
"trainable_variables
Fmetrics
#	variables

Glayers
Hnon_trainable_variables
$regularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"Ώ
_tf_keras_metric€{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
,:*
¬2Adam/hiddenLayer1/kernel/m
%:#¬2Adam/hiddenLayer1/bias/m
+:)	¬d2Adam/hiddenLayer2/kernel/m
$:"d2Adam/hiddenLayer2/bias/m
):'d
2Adam/outputLayer/kernel/m
#:!
2Adam/outputLayer/bias/m
,:*
¬2Adam/hiddenLayer1/kernel/v
%:#¬2Adam/hiddenLayer1/bias/v
+:)	¬d2Adam/hiddenLayer2/kernel/v
$:"d2Adam/hiddenLayer2/bias/v
):'d
2Adam/outputLayer/kernel/v
#:!
2Adam/outputLayer/bias/v
φ2σ
*__inference_sequential_layer_call_fn_39052
*__inference_sequential_layer_call_fn_39197
*__inference_sequential_layer_call_fn_39214
*__inference_sequential_layer_call_fn_39090ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
E__inference_sequential_layer_call_and_return_conditional_losses_39152
E__inference_sequential_layer_call_and_return_conditional_losses_39180
E__inference_sequential_layer_call_and_return_conditional_losses_38992
E__inference_sequential_layer_call_and_return_conditional_losses_39013ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
λ2θ
 __inference__wrapped_model_38862Γ
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
annotationsͺ *3’0
.+
inputLayer_input?????????
Τ2Ρ
*__inference_inputLayer_layer_call_fn_39225’
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
annotationsͺ *
 
ο2μ
E__inference_inputLayer_layer_call_and_return_conditional_losses_39220’
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
annotationsͺ *
 
Φ2Σ
,__inference_hiddenLayer1_layer_call_fn_39245’
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
annotationsͺ *
 
ρ2ξ
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39236’
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
annotationsͺ *
 
2
'__inference_dropout_layer_call_fn_39272
'__inference_dropout_layer_call_fn_39267΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Β2Ώ
B__inference_dropout_layer_call_and_return_conditional_losses_39257
B__inference_dropout_layer_call_and_return_conditional_losses_39262΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
,__inference_hiddenLayer2_layer_call_fn_39292’
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
annotationsͺ *
 
ρ2ξ
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39283’
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
annotationsͺ *
 
Υ2?
+__inference_outputLayer_layer_call_fn_39312’
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
annotationsͺ *
 
π2ν
F__inference_outputLayer_layer_call_and_return_conditional_losses_39303’
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
annotationsͺ *
 
;B9
#__inference_signature_wrapper_39117inputLayer_input§
 __inference__wrapped_model_38862 !=’:
3’0
.+
inputLayer_input?????????
ͺ "9ͺ6
4
outputLayer%"
outputLayer?????????
€
B__inference_dropout_layer_call_and_return_conditional_losses_39257^4’1
*’'
!
inputs?????????¬
p
ͺ "&’#

0?????????¬
 €
B__inference_dropout_layer_call_and_return_conditional_losses_39262^4’1
*’'
!
inputs?????????¬
p 
ͺ "&’#

0?????????¬
 |
'__inference_dropout_layer_call_fn_39267Q4’1
*’'
!
inputs?????????¬
p
ͺ "?????????¬|
'__inference_dropout_layer_call_fn_39272Q4’1
*’'
!
inputs?????????¬
p 
ͺ "?????????¬©
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39236^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????¬
 
,__inference_hiddenLayer1_layer_call_fn_39245Q0’-
&’#
!
inputs?????????
ͺ "?????????¬¨
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39283]0’-
&’#
!
inputs?????????¬
ͺ "%’"

0?????????d
 
,__inference_hiddenLayer2_layer_call_fn_39292P0’-
&’#
!
inputs?????????¬
ͺ "?????????d¦
E__inference_inputLayer_layer_call_and_return_conditional_losses_39220]3’0
)’&
$!
inputs?????????
ͺ "&’#

0?????????
 ~
*__inference_inputLayer_layer_call_fn_39225P3’0
)’&
$!
inputs?????????
ͺ "?????????¦
F__inference_outputLayer_layer_call_and_return_conditional_losses_39303\ !/’,
%’"
 
inputs?????????d
ͺ "%’"

0?????????

 ~
+__inference_outputLayer_layer_call_fn_39312O !/’,
%’"
 
inputs?????????d
ͺ "?????????
Ώ
E__inference_sequential_layer_call_and_return_conditional_losses_38992v !E’B
;’8
.+
inputLayer_input?????????
p

 
ͺ "%’"

0?????????

 Ώ
E__inference_sequential_layer_call_and_return_conditional_losses_39013v !E’B
;’8
.+
inputLayer_input?????????
p 

 
ͺ "%’"

0?????????

 ΅
E__inference_sequential_layer_call_and_return_conditional_losses_39152l !;’8
1’.
$!
inputs?????????
p

 
ͺ "%’"

0?????????

 ΅
E__inference_sequential_layer_call_and_return_conditional_losses_39180l !;’8
1’.
$!
inputs?????????
p 

 
ͺ "%’"

0?????????

 
*__inference_sequential_layer_call_fn_39052i !E’B
;’8
.+
inputLayer_input?????????
p

 
ͺ "?????????

*__inference_sequential_layer_call_fn_39090i !E’B
;’8
.+
inputLayer_input?????????
p 

 
ͺ "?????????

*__inference_sequential_layer_call_fn_39197_ !;’8
1’.
$!
inputs?????????
p

 
ͺ "?????????

*__inference_sequential_layer_call_fn_39214_ !;’8
1’.
$!
inputs?????????
p 

 
ͺ "?????????
Ύ
#__inference_signature_wrapper_39117 !Q’N
’ 
GͺD
B
inputLayer_input.+
inputLayer_input?????????"9ͺ6
4
outputLayer%"
outputLayer?????????
