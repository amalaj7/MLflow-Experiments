??

??
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
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??
?
hiddenLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namehiddenLayer1/kernel
}
'hiddenLayer1/kernel/Read/ReadVariableOpReadVariableOphiddenLayer1/kernel* 
_output_shapes
:
??*
dtype0
{
hiddenLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehiddenLayer1/bias
t
%hiddenLayer1/bias/Read/ReadVariableOpReadVariableOphiddenLayer1/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
hiddenLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namehiddenLayer2/kernel
}
'hiddenLayer2/kernel/Read/ReadVariableOpReadVariableOphiddenLayer2/kernel* 
_output_shapes
:
??*
dtype0
{
hiddenLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namehiddenLayer2/bias
t
%hiddenLayer2/bias/Read/ReadVariableOpReadVariableOphiddenLayer2/bias*
_output_shapes	
:?*
dtype0
?
hiddenLayer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*$
shared_namehiddenLayer3/kernel
|
'hiddenLayer3/kernel/Read/ReadVariableOpReadVariableOphiddenLayer3/kernel*
_output_shapes
:	?d*
dtype0
z
hiddenLayer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namehiddenLayer3/bias
s
%hiddenLayer3/bias/Read/ReadVariableOpReadVariableOphiddenLayer3/bias*
_output_shapes
:d*
dtype0
?
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
?
Adam/hiddenLayer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/hiddenLayer1/kernel/m
?
.Adam/hiddenLayer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/hiddenLayer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/hiddenLayer1/bias/m
?
,Adam/hiddenLayer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/hiddenLayer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/hiddenLayer2/kernel/m
?
.Adam/hiddenLayer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/hiddenLayer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/hiddenLayer2/bias/m
?
,Adam/hiddenLayer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/hiddenLayer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*+
shared_nameAdam/hiddenLayer3/kernel/m
?
.Adam/hiddenLayer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer3/kernel/m*
_output_shapes
:	?d*
dtype0
?
Adam/hiddenLayer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hiddenLayer3/bias/m
?
,Adam/hiddenLayer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer3/bias/m*
_output_shapes
:d*
dtype0
?
Adam/outputLayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
**
shared_nameAdam/outputLayer/kernel/m
?
-Adam/outputLayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/outputLayer/kernel/m*
_output_shapes

:d
*
dtype0
?
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
?
Adam/hiddenLayer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/hiddenLayer1/kernel/v
?
.Adam/hiddenLayer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/hiddenLayer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/hiddenLayer1/bias/v
?
,Adam/hiddenLayer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer1/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/hiddenLayer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*+
shared_nameAdam/hiddenLayer2/kernel/v
?
.Adam/hiddenLayer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/hiddenLayer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/hiddenLayer2/bias/v
?
,Adam/hiddenLayer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/hiddenLayer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*+
shared_nameAdam/hiddenLayer3/kernel/v
?
.Adam/hiddenLayer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer3/kernel/v*
_output_shapes
:	?d*
dtype0
?
Adam/hiddenLayer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*)
shared_nameAdam/hiddenLayer3/bias/v
?
,Adam/hiddenLayer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddenLayer3/bias/v*
_output_shapes
:d*
dtype0
?
Adam/outputLayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
**
shared_nameAdam/outputLayer/kernel/v
?
-Adam/outputLayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/outputLayer/kernel/v*
_output_shapes

:d
*
dtype0
?
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
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemompmqmr!ms"mt+mu,mv1mw2mxvyvzv{v|!v}"v~+v,v?1v?2v?
 
F
0
1
2
3
!4
"5
+6
,7
18
29
V
0
1
2
3
4
5
!6
"7
+8
,9
110
211
?

<layers
	regularization_losses
=layer_metrics
>layer_regularization_losses
?non_trainable_variables

trainable_variables
	variables
@metrics
 
 
 
 
?

Alayers
Blayer_metrics
regularization_losses
Cmetrics
trainable_variables
	variables
Dnon_trainable_variables
Elayer_regularization_losses
_]
VARIABLE_VALUEhiddenLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Flayers
Glayer_metrics
regularization_losses
Hmetrics
trainable_variables
	variables
Inon_trainable_variables
Jlayer_regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
?

Klayers
Llayer_metrics
regularization_losses
Mmetrics
trainable_variables
	variables
Nnon_trainable_variables
Olayer_regularization_losses
_]
VARIABLE_VALUEhiddenLayer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?

Players
Qlayer_metrics
#regularization_losses
Rmetrics
$trainable_variables
%	variables
Snon_trainable_variables
Tlayer_regularization_losses
 
 
 
?

Ulayers
Vlayer_metrics
'regularization_losses
Wmetrics
(trainable_variables
)	variables
Xnon_trainable_variables
Ylayer_regularization_losses
_]
VARIABLE_VALUEhiddenLayer3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEhiddenLayer3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?

Zlayers
[layer_metrics
-regularization_losses
\metrics
.trainable_variables
/	variables
]non_trainable_variables
^layer_regularization_losses
^\
VARIABLE_VALUEoutputLayer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEoutputLayer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?

_layers
`layer_metrics
3regularization_losses
ametrics
4trainable_variables
5	variables
bnon_trainable_variables
clayer_regularization_losses
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
1
0
1
2
3
4
5
6
 
 

0
1

d0
e1
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

0
1
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
	ftotal
	gcount
h	variables
i	keras_api
D
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

m	variables
??
VARIABLE_VALUEAdam/hiddenLayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hiddenLayer2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hiddenLayer3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/outputLayer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/outputLayer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hiddenLayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hiddenLayer2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hiddenLayer3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hiddenLayer3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/outputLayer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/outputLayer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_inputLayer_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_inputLayer_inputhiddenLayer1/kernelhiddenLayer1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betahiddenLayer2/kernelhiddenLayer2/biashiddenLayer3/kernelhiddenLayer3/biasoutputLayer/kerneloutputLayer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_39836
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'hiddenLayer1/kernel/Read/ReadVariableOp%hiddenLayer1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp'hiddenLayer2/kernel/Read/ReadVariableOp%hiddenLayer2/bias/Read/ReadVariableOp'hiddenLayer3/kernel/Read/ReadVariableOp%hiddenLayer3/bias/Read/ReadVariableOp&outputLayer/kernel/Read/ReadVariableOp$outputLayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/hiddenLayer1/kernel/m/Read/ReadVariableOp,Adam/hiddenLayer1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp.Adam/hiddenLayer2/kernel/m/Read/ReadVariableOp,Adam/hiddenLayer2/bias/m/Read/ReadVariableOp.Adam/hiddenLayer3/kernel/m/Read/ReadVariableOp,Adam/hiddenLayer3/bias/m/Read/ReadVariableOp-Adam/outputLayer/kernel/m/Read/ReadVariableOp+Adam/outputLayer/bias/m/Read/ReadVariableOp.Adam/hiddenLayer1/kernel/v/Read/ReadVariableOp,Adam/hiddenLayer1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp.Adam/hiddenLayer2/kernel/v/Read/ReadVariableOp,Adam/hiddenLayer2/bias/v/Read/ReadVariableOp.Adam/hiddenLayer3/kernel/v/Read/ReadVariableOp,Adam/hiddenLayer3/bias/v/Read/ReadVariableOp-Adam/outputLayer/kernel/v/Read/ReadVariableOp+Adam/outputLayer/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_40362
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehiddenLayer1/kernelhiddenLayer1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancehiddenLayer2/kernelhiddenLayer2/biashiddenLayer3/kernelhiddenLayer3/biasoutputLayer/kerneloutputLayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hiddenLayer1/kernel/mAdam/hiddenLayer1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/hiddenLayer2/kernel/mAdam/hiddenLayer2/bias/mAdam/hiddenLayer3/kernel/mAdam/hiddenLayer3/bias/mAdam/outputLayer/kernel/mAdam/outputLayer/bias/mAdam/hiddenLayer1/kernel/vAdam/hiddenLayer1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/hiddenLayer2/kernel/vAdam/hiddenLayer2/bias/vAdam/hiddenLayer3/kernel/vAdam/hiddenLayer3/bias/vAdam/outputLayer/kernel/vAdam/outputLayer/bias/v*5
Tin.
,2**
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_40495??
?&
?
E__inference_sequential_layer_call_and_return_conditional_losses_39633
inputlayer_input
hiddenlayer1_39481
hiddenlayer1_39483
batch_normalization_39512
batch_normalization_39514
batch_normalization_39516
batch_normalization_39518
hiddenlayer2_39543
hiddenlayer2_39545
hiddenlayer3_39600
hiddenlayer3_39602
outputlayer_39627
outputlayer_39629
identity??+batch_normalization/StatefulPartitionedCall?dropout/StatefulPartitionedCall?$hiddenLayer1/StatefulPartitionedCall?$hiddenLayer2/StatefulPartitionedCall?$hiddenLayer3/StatefulPartitionedCall?#outputLayer/StatefulPartitionedCall?
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_394522
inputLayer/PartitionedCall?
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39481hiddenlayer1_39483*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_394702&
$hiddenLayer1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0batch_normalization_39512batch_normalization_39514batch_normalization_39516batch_normalization_39518*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_393982-
+batch_normalization/StatefulPartitionedCall?
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0hiddenlayer2_39543hiddenlayer2_39545*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395322&
$hiddenLayer2/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395602!
dropout/StatefulPartitionedCall?
$hiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hiddenlayer3_39600hiddenlayer3_39602*
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
GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_395892&
$hiddenLayer3/StatefulPartitionedCall?
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer3/StatefulPartitionedCall:output:0outputlayer_39627outputlayer_39629*
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
GPU2*0J 8? *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_396162%
#outputLayer/StatefulPartitionedCall?
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall%^hiddenLayer3/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2L
$hiddenLayer3/StatefulPartitionedCall$hiddenLayer3/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?
?
+__inference_outputLayer_layer_call_fn_40216

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_396162
StatefulPartitionedCall?
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
?
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_39452

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
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_39532

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_40187

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
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
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_39589

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
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
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_39836
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_393022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?
?
F__inference_outputLayer_layer_call_and_return_conditional_losses_40207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_39431

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
E__inference_sequential_layer_call_and_return_conditional_losses_39959

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource/
+hiddenlayer3_matmul_readvariableop_resource0
,hiddenlayer3_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identity?u
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
inputLayer/Const?
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:??????????2
inputLayer/Reshape?
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp?
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer1/MatMul?
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOp?
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer1/BiasAdd?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1MulhiddenLayer1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/mul_1?
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/add_1?
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp?
hiddenLayer2/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/MatMul?
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOp?
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/BiasAdd?
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/Relu?
dropout/IdentityIdentityhiddenLayer2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
"hiddenLayer3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"hiddenLayer3/MatMul/ReadVariableOp?
hiddenLayer3/MatMulMatMuldropout/Identity:output:0*hiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/MatMul?
#hiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer3/BiasAdd/ReadVariableOp?
hiddenLayer3/BiasAddBiasAddhiddenLayer3/MatMul:product:0+hiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/BiasAdd
hiddenLayer3/ReluReluhiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/Relu?
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp?
outputLayer/MatMulMatMulhiddenLayer3/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/MatMul?
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp?
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/BiasAdd?
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
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_inputLayer_layer_call_and_return_conditional_losses_40023

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
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_39988

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_397062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40103

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_40176

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_40038

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_hiddenLayer3_layer_call_fn_40196

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_395892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_39470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
__inference__traced_save_40362
file_prefix2
.savev2_hiddenlayer1_kernel_read_readvariableop0
,savev2_hiddenlayer1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop2
.savev2_hiddenlayer2_kernel_read_readvariableop0
,savev2_hiddenlayer2_bias_read_readvariableop2
.savev2_hiddenlayer3_kernel_read_readvariableop0
,savev2_hiddenlayer3_bias_read_readvariableop1
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
3savev2_adam_hiddenlayer1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer3_kernel_m_read_readvariableop7
3savev2_adam_hiddenlayer3_bias_m_read_readvariableop8
4savev2_adam_outputlayer_kernel_m_read_readvariableop6
2savev2_adam_outputlayer_bias_m_read_readvariableop9
5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop9
5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer2_bias_v_read_readvariableop9
5savev2_adam_hiddenlayer3_kernel_v_read_readvariableop7
3savev2_adam_hiddenlayer3_bias_v_read_readvariableop8
4savev2_adam_outputlayer_kernel_v_read_readvariableop6
2savev2_adam_outputlayer_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b8e9cc9d4dce4287be14d511d8e94912/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddenlayer1_kernel_read_readvariableop,savev2_hiddenlayer1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop.savev2_hiddenlayer2_kernel_read_readvariableop,savev2_hiddenlayer2_bias_read_readvariableop.savev2_hiddenlayer3_kernel_read_readvariableop,savev2_hiddenlayer3_bias_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_hiddenlayer1_kernel_m_read_readvariableop3savev2_adam_hiddenlayer1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop5savev2_adam_hiddenlayer2_kernel_m_read_readvariableop3savev2_adam_hiddenlayer2_bias_m_read_readvariableop5savev2_adam_hiddenlayer3_kernel_m_read_readvariableop3savev2_adam_hiddenlayer3_bias_m_read_readvariableop4savev2_adam_outputlayer_kernel_m_read_readvariableop2savev2_adam_outputlayer_bias_m_read_readvariableop5savev2_adam_hiddenlayer1_kernel_v_read_readvariableop3savev2_adam_hiddenlayer1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop5savev2_adam_hiddenlayer2_kernel_v_read_readvariableop3savev2_adam_hiddenlayer2_bias_v_read_readvariableop5savev2_adam_hiddenlayer3_kernel_v_read_readvariableop3savev2_adam_hiddenlayer3_bias_v_read_readvariableop4savev2_adam_outputlayer_kernel_v_read_readvariableop2savev2_adam_outputlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:?:?:?:?:
??:?:	?d:d:d
:
: : : : : : : : : :
??:?:?:?:
??:?:	?d:d:d
:
:
??:?:?:?:
??:?:	?d:d:d
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
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?d: 


_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:!"

_output_shapes	
:?:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?d: '

_output_shapes
:d:$( 

_output_shapes

:d
: )

_output_shapes
:
:*

_output_shapes
: 
?C
?
 __inference__wrapped_model_39302
inputlayer_input:
6sequential_hiddenlayer1_matmul_readvariableop_resource;
7sequential_hiddenlayer1_biasadd_readvariableop_resourceD
@sequential_batch_normalization_batchnorm_readvariableop_resourceH
Dsequential_batch_normalization_batchnorm_mul_readvariableop_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_1_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_2_resource:
6sequential_hiddenlayer2_matmul_readvariableop_resource;
7sequential_hiddenlayer2_biasadd_readvariableop_resource:
6sequential_hiddenlayer3_matmul_readvariableop_resource;
7sequential_hiddenlayer3_biasadd_readvariableop_resource9
5sequential_outputlayer_matmul_readvariableop_resource:
6sequential_outputlayer_biasadd_readvariableop_resource
identity??
sequential/inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
sequential/inputLayer/Const?
sequential/inputLayer/ReshapeReshapeinputlayer_input$sequential/inputLayer/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/inputLayer/Reshape?
-sequential/hiddenLayer1/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential/hiddenLayer1/MatMul/ReadVariableOp?
sequential/hiddenLayer1/MatMulMatMul&sequential/inputLayer/Reshape:output:05sequential/hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential/hiddenLayer1/MatMul?
.sequential/hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential/hiddenLayer1/BiasAdd/ReadVariableOp?
sequential/hiddenLayer1/BiasAddBiasAdd(sequential/hiddenLayer1/MatMul:product:06sequential/hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential/hiddenLayer1/BiasAdd?
7sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/batch_normalization/batchnorm/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2?sequential/batch_normalization/batchnorm/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDsequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0Csequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mul(sequential/hiddenLayer1/BiasAdd:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/mul_1?
9sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_1?
.sequential/batch_normalization/batchnorm/mul_2MulAsequential/batch_normalization/batchnorm/ReadVariableOp_1:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
9sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_2?
,sequential/batch_normalization/batchnorm/subSubAsequential/batch_normalization/batchnorm/ReadVariableOp_2:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.sequential/batch_normalization/batchnorm/add_1?
-sequential/hiddenLayer2/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-sequential/hiddenLayer2/MatMul/ReadVariableOp?
sequential/hiddenLayer2/MatMulMatMul2sequential/batch_normalization/batchnorm/add_1:z:05sequential/hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential/hiddenLayer2/MatMul?
.sequential/hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential/hiddenLayer2/BiasAdd/ReadVariableOp?
sequential/hiddenLayer2/BiasAddBiasAdd(sequential/hiddenLayer2/MatMul:product:06sequential/hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential/hiddenLayer2/BiasAdd?
sequential/hiddenLayer2/ReluRelu(sequential/hiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/hiddenLayer2/Relu?
sequential/dropout/IdentityIdentity*sequential/hiddenLayer2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
sequential/dropout/Identity?
-sequential/hiddenLayer3/MatMul/ReadVariableOpReadVariableOp6sequential_hiddenlayer3_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02/
-sequential/hiddenLayer3/MatMul/ReadVariableOp?
sequential/hiddenLayer3/MatMulMatMul$sequential/dropout/Identity:output:05sequential/hiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential/hiddenLayer3/MatMul?
.sequential/hiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp7sequential_hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype020
.sequential/hiddenLayer3/BiasAdd/ReadVariableOp?
sequential/hiddenLayer3/BiasAddBiasAdd(sequential/hiddenLayer3/MatMul:product:06sequential/hiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential/hiddenLayer3/BiasAdd?
sequential/hiddenLayer3/ReluRelu(sequential/hiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential/hiddenLayer3/Relu?
,sequential/outputLayer/MatMul/ReadVariableOpReadVariableOp5sequential_outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02.
,sequential/outputLayer/MatMul/ReadVariableOp?
sequential/outputLayer/MatMulMatMul*sequential/hiddenLayer3/Relu:activations:04sequential/outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/outputLayer/MatMul?
-sequential/outputLayer/BiasAdd/ReadVariableOpReadVariableOp6sequential_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential/outputLayer/BiasAdd/ReadVariableOp?
sequential/outputLayer/BiasAddBiasAdd'sequential/outputLayer/MatMul:product:05sequential/outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential/outputLayer/BiasAdd?
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
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_40166

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_39733
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_397062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?d
?
E__inference_sequential_layer_call_and_return_conditional_losses_39909

inputs/
+hiddenlayer1_matmul_readvariableop_resource0
,hiddenlayer1_biasadd_readvariableop_resource-
)batch_normalization_assignmovingavg_39855/
+batch_normalization_assignmovingavg_1_39861=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource/
+hiddenlayer2_matmul_readvariableop_resource0
,hiddenlayer2_biasadd_readvariableop_resource/
+hiddenlayer3_matmul_readvariableop_resource0
,hiddenlayer3_biasadd_readvariableop_resource.
*outputlayer_matmul_readvariableop_resource/
+outputlayer_biasadd_readvariableop_resource
identity??7batch_normalization/AssignMovingAvg/AssignSubVariableOp?9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpu
inputLayer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
inputLayer/Const?
inputLayer/ReshapeReshapeinputsinputLayer/Const:output:0*
T0*(
_output_shapes
:??????????2
inputLayer/Reshape?
"hiddenLayer1/MatMul/ReadVariableOpReadVariableOp+hiddenlayer1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hiddenLayer1/MatMul/ReadVariableOp?
hiddenLayer1/MatMulMatMulinputLayer/Reshape:output:0*hiddenLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer1/MatMul?
#hiddenLayer1/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hiddenLayer1/BiasAdd/ReadVariableOp?
hiddenLayer1/BiasAddBiasAddhiddenLayer1/MatMul:product:0+hiddenLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer1/BiasAdd?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeanhiddenLayer1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	?2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferencehiddenLayer1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/39855*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_39855*
_output_shapes	
:?*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/39855*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/39855*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/mul?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_39855+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/39855*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp?
+batch_normalization/AssignMovingAvg_1/decayConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/39861*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_39861*
_output_shapes	
:?*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/39861*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/39861*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/mul?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_39861-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/39861*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1MulhiddenLayer1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2%
#batch_normalization/batchnorm/add_1?
"hiddenLayer2/MatMul/ReadVariableOpReadVariableOp+hiddenlayer2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hiddenLayer2/MatMul/ReadVariableOp?
hiddenLayer2/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0*hiddenLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/MatMul?
#hiddenLayer2/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hiddenLayer2/BiasAdd/ReadVariableOp?
hiddenLayer2/BiasAddBiasAddhiddenLayer2/MatMul:product:0+hiddenLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/BiasAdd?
hiddenLayer2/ReluReluhiddenLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hiddenLayer2/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulhiddenLayer2/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul}
dropout/dropout/ShapeShapehiddenLayer2/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
"hiddenLayer3/MatMul/ReadVariableOpReadVariableOp+hiddenlayer3_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02$
"hiddenLayer3/MatMul/ReadVariableOp?
hiddenLayer3/MatMulMatMuldropout/dropout/Mul_1:z:0*hiddenLayer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/MatMul?
#hiddenLayer3/BiasAdd/ReadVariableOpReadVariableOp,hiddenlayer3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#hiddenLayer3/BiasAdd/ReadVariableOp?
hiddenLayer3/BiasAddBiasAddhiddenLayer3/MatMul:product:0+hiddenLayer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/BiasAdd
hiddenLayer3/ReluReluhiddenLayer3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
hiddenLayer3/Relu?
!outputLayer/MatMul/ReadVariableOpReadVariableOp*outputlayer_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype02#
!outputLayer/MatMul/ReadVariableOp?
outputLayer/MatMulMatMulhiddenLayer3/Relu:activations:0)outputLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/MatMul?
"outputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"outputLayer/BiasAdd/ReadVariableOp?
outputLayer/BiasAddBiasAddoutputLayer/MatMul:product:0*outputLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
outputLayer/BiasAdd?
outputLayer/SoftmaxSoftmaxoutputLayer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
outputLayer/Softmax?
IdentityIdentityoutputLayer/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_39797
inputlayer_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputlayer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_397702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?$
?
E__inference_sequential_layer_call_and_return_conditional_losses_39668
inputlayer_input
hiddenlayer1_39637
hiddenlayer1_39639
batch_normalization_39642
batch_normalization_39644
batch_normalization_39646
batch_normalization_39648
hiddenlayer2_39651
hiddenlayer2_39653
hiddenlayer3_39657
hiddenlayer3_39659
outputlayer_39662
outputlayer_39664
identity??+batch_normalization/StatefulPartitionedCall?$hiddenLayer1/StatefulPartitionedCall?$hiddenLayer2/StatefulPartitionedCall?$hiddenLayer3/StatefulPartitionedCall?#outputLayer/StatefulPartitionedCall?
inputLayer/PartitionedCallPartitionedCallinputlayer_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_394522
inputLayer/PartitionedCall?
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39637hiddenlayer1_39639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_394702&
$hiddenLayer1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0batch_normalization_39642batch_normalization_39644batch_normalization_39646batch_normalization_39648*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_394312-
+batch_normalization/StatefulPartitionedCall?
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0hiddenlayer2_39651hiddenlayer2_39653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395322&
$hiddenLayer2/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395652
dropout/PartitionedCall?
$hiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hiddenlayer3_39657hiddenlayer3_39659*
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
GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_395892&
$hiddenLayer3/StatefulPartitionedCall?
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer3/StatefulPartitionedCall:output:0outputlayer_39662outputlayer_39664*
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
GPU2*0J 8? *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_396162%
#outputLayer/StatefulPartitionedCall?
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall%^hiddenLayer3/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2L
$hiddenLayer3/StatefulPartitionedCall$hiddenLayer3/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:] Y
+
_output_shapes
:?????????
*
_user_specified_nameinputLayer_input
?
?
F__inference_outputLayer_layer_call_and_return_conditional_losses_39616

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
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
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
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
?
`
'__inference_dropout_layer_call_fn_40171

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_hiddenLayer1_layer_call_fn_40047

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_394702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_39565

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
E__inference_sequential_layer_call_and_return_conditional_losses_39706

inputs
hiddenlayer1_39675
hiddenlayer1_39677
batch_normalization_39680
batch_normalization_39682
batch_normalization_39684
batch_normalization_39686
hiddenlayer2_39689
hiddenlayer2_39691
hiddenlayer3_39695
hiddenlayer3_39697
outputlayer_39700
outputlayer_39702
identity??+batch_normalization/StatefulPartitionedCall?dropout/StatefulPartitionedCall?$hiddenLayer1/StatefulPartitionedCall?$hiddenLayer2/StatefulPartitionedCall?$hiddenLayer3/StatefulPartitionedCall?#outputLayer/StatefulPartitionedCall?
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_394522
inputLayer/PartitionedCall?
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39675hiddenlayer1_39677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_394702&
$hiddenLayer1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0batch_normalization_39680batch_normalization_39682batch_normalization_39684batch_normalization_39686*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_393982-
+batch_normalization/StatefulPartitionedCall?
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0hiddenlayer2_39689hiddenlayer2_39691*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395322&
$hiddenLayer2/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395602!
dropout/StatefulPartitionedCall?
$hiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hiddenlayer3_39695hiddenlayer3_39697*
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
GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_395892&
$hiddenLayer3/StatefulPartitionedCall?
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer3/StatefulPartitionedCall:output:0outputlayer_39700outputlayer_39702*
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
GPU2*0J 8? *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_396162%
#outputLayer/StatefulPartitionedCall?
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall ^dropout/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall%^hiddenLayer3/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2L
$hiddenLayer3/StatefulPartitionedCall$hiddenLayer3/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_40161

inputs
identity?c
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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40083

inputs
assignmovingavg_40058
assignmovingavg_1_40064)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/40058*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_40058*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/40058*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/40058*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_40058AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/40058*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/40064*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_40064*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/40064*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/40064*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_40064AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/40064*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_39560

inputs
identity?c
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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_inputLayer_layer_call_fn_40028

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_394522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_40140

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_40129

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_394312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_40116

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_393982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_40017

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_397702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_40495
file_prefix(
$assignvariableop_hiddenlayer1_kernel(
$assignvariableop_1_hiddenlayer1_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance*
&assignvariableop_6_hiddenlayer2_kernel(
$assignvariableop_7_hiddenlayer2_bias*
&assignvariableop_8_hiddenlayer3_kernel(
$assignvariableop_9_hiddenlayer3_bias*
&assignvariableop_10_outputlayer_kernel(
$assignvariableop_11_outputlayer_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_12
.assignvariableop_21_adam_hiddenlayer1_kernel_m0
,assignvariableop_22_adam_hiddenlayer1_bias_m8
4assignvariableop_23_adam_batch_normalization_gamma_m7
3assignvariableop_24_adam_batch_normalization_beta_m2
.assignvariableop_25_adam_hiddenlayer2_kernel_m0
,assignvariableop_26_adam_hiddenlayer2_bias_m2
.assignvariableop_27_adam_hiddenlayer3_kernel_m0
,assignvariableop_28_adam_hiddenlayer3_bias_m1
-assignvariableop_29_adam_outputlayer_kernel_m/
+assignvariableop_30_adam_outputlayer_bias_m2
.assignvariableop_31_adam_hiddenlayer1_kernel_v0
,assignvariableop_32_adam_hiddenlayer1_bias_v8
4assignvariableop_33_adam_batch_normalization_gamma_v7
3assignvariableop_34_adam_batch_normalization_beta_v2
.assignvariableop_35_adam_hiddenlayer2_kernel_v0
,assignvariableop_36_adam_hiddenlayer2_bias_v2
.assignvariableop_37_adam_hiddenlayer3_kernel_v0
,assignvariableop_38_adam_hiddenlayer3_bias_v1
-assignvariableop_39_adam_outputlayer_kernel_v/
+assignvariableop_40_adam_outputlayer_bias_v
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_hiddenlayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddenlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_hiddenlayer2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_hiddenlayer2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_hiddenlayer3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_hiddenlayer3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_outputlayer_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_outputlayer_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_hiddenlayer1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_hiddenlayer1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_batch_normalization_gamma_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_batch_normalization_beta_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_hiddenlayer2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_hiddenlayer2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_hiddenlayer3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_hiddenlayer3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_outputlayer_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_outputlayer_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_hiddenlayer1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_hiddenlayer1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_batch_normalization_gamma_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_batch_normalization_beta_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_hiddenlayer2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_hiddenlayer2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_hiddenlayer3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_hiddenlayer3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_outputlayer_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_outputlayer_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
?)
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_39398

inputs
assignmovingavg_39373
assignmovingavg_1_39379)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/39373*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_39373*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/39373*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/39373*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_39373AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/39373*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/39379*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_39379*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/39379*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/39379*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_39379AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/39379*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_hiddenLayer2_layer_call_fn_40149

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
E__inference_sequential_layer_call_and_return_conditional_losses_39770

inputs
hiddenlayer1_39739
hiddenlayer1_39741
batch_normalization_39744
batch_normalization_39746
batch_normalization_39748
batch_normalization_39750
hiddenlayer2_39753
hiddenlayer2_39755
hiddenlayer3_39759
hiddenlayer3_39761
outputlayer_39764
outputlayer_39766
identity??+batch_normalization/StatefulPartitionedCall?$hiddenLayer1/StatefulPartitionedCall?$hiddenLayer2/StatefulPartitionedCall?$hiddenLayer3/StatefulPartitionedCall?#outputLayer/StatefulPartitionedCall?
inputLayer/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_inputLayer_layer_call_and_return_conditional_losses_394522
inputLayer/PartitionedCall?
$hiddenLayer1/StatefulPartitionedCallStatefulPartitionedCall#inputLayer/PartitionedCall:output:0hiddenlayer1_39739hiddenlayer1_39741*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_394702&
$hiddenLayer1/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer1/StatefulPartitionedCall:output:0batch_normalization_39744batch_normalization_39746batch_normalization_39748batch_normalization_39750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_394312-
+batch_normalization/StatefulPartitionedCall?
$hiddenLayer2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0hiddenlayer2_39753hiddenlayer2_39755*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_395322&
$hiddenLayer2/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall-hiddenLayer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_395652
dropout/PartitionedCall?
$hiddenLayer3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hiddenlayer3_39759hiddenlayer3_39761*
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
GPU2*0J 8? *P
fKRI
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_395892&
$hiddenLayer3/StatefulPartitionedCall?
#outputLayer/StatefulPartitionedCallStatefulPartitionedCall-hiddenLayer3/StatefulPartitionedCall:output:0outputlayer_39764outputlayer_39766*
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
GPU2*0J 8? *O
fJRH
F__inference_outputLayer_layer_call_and_return_conditional_losses_396162%
#outputLayer/StatefulPartitionedCall?
IdentityIdentity,outputLayer/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall%^hiddenLayer1/StatefulPartitionedCall%^hiddenLayer2/StatefulPartitionedCall%^hiddenLayer3/StatefulPartitionedCall$^outputLayer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$hiddenLayer1/StatefulPartitionedCall$hiddenLayer1/StatefulPartitionedCall2L
$hiddenLayer2/StatefulPartitionedCall$hiddenLayer2/StatefulPartitionedCall2L
$hiddenLayer3/StatefulPartitionedCall$hiddenLayer3/StatefulPartitionedCall2J
#outputLayer/StatefulPartitionedCall#outputLayer/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
inputLayer_input=
"serving_default_inputLayer_input:0??????????
outputLayer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?7
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?4
_tf_keras_sequential?3{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 350, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputLayer_input"}}, {"class_name": "Flatten", "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 350, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hiddenLayer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "inputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "inputLayer", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hiddenLayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hiddenLayer1", "trainable": true, "dtype": "float32", "units": 350, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
?	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 350}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350]}}
?

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hiddenLayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hiddenLayer2", "trainable": true, "dtype": "float32", "units": 300, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 350}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350]}}
?
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hiddenLayer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hiddenLayer3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "outputLayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "outputLayer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemompmqmr!ms"mt+mu,mv1mw2mxvyvzv{v|!v}"v~+v,v?1v?2v?"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
!4
"5
+6
,7
18
29"
trackable_list_wrapper
v
0
1
2
3
4
5
!6
"7
+8
,9
110
211"
trackable_list_wrapper
?

<layers
	regularization_losses
=layer_metrics
>layer_regularization_losses
?non_trainable_variables

trainable_variables
	variables
@metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Alayers
Blayer_metrics
regularization_losses
Cmetrics
trainable_variables
	variables
Dnon_trainable_variables
Elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%
??2hiddenLayer1/kernel
 :?2hiddenLayer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Flayers
Glayer_metrics
regularization_losses
Hmetrics
trainable_variables
	variables
Inon_trainable_variables
Jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?

Klayers
Llayer_metrics
regularization_losses
Mmetrics
trainable_variables
	variables
Nnon_trainable_variables
Olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%
??2hiddenLayer2/kernel
 :?2hiddenLayer2/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?

Players
Qlayer_metrics
#regularization_losses
Rmetrics
$trainable_variables
%	variables
Snon_trainable_variables
Tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ulayers
Vlayer_metrics
'regularization_losses
Wmetrics
(trainable_variables
)	variables
Xnon_trainable_variables
Ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?d2hiddenLayer3/kernel
:d2hiddenLayer3/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?

Zlayers
[layer_metrics
-regularization_losses
\metrics
.trainable_variables
/	variables
]non_trainable_variables
^layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"d
2outputLayer/kernel
:
2outputLayer/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?

_layers
`layer_metrics
3regularization_losses
ametrics
4trainable_variables
5	variables
bnon_trainable_variables
clayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
d0
e1"
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
.
0
1"
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
?
	ftotal
	gcount
h	variables
i	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	jtotal
	kcount
l
_fn_kwargs
m	variables
n	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
,:*
??2Adam/hiddenLayer1/kernel/m
%:#?2Adam/hiddenLayer1/bias/m
-:+?2 Adam/batch_normalization/gamma/m
,:*?2Adam/batch_normalization/beta/m
,:*
??2Adam/hiddenLayer2/kernel/m
%:#?2Adam/hiddenLayer2/bias/m
+:)	?d2Adam/hiddenLayer3/kernel/m
$:"d2Adam/hiddenLayer3/bias/m
):'d
2Adam/outputLayer/kernel/m
#:!
2Adam/outputLayer/bias/m
,:*
??2Adam/hiddenLayer1/kernel/v
%:#?2Adam/hiddenLayer1/bias/v
-:+?2 Adam/batch_normalization/gamma/v
,:*?2Adam/batch_normalization/beta/v
,:*
??2Adam/hiddenLayer2/kernel/v
%:#?2Adam/hiddenLayer2/bias/v
+:)	?d2Adam/hiddenLayer3/kernel/v
$:"d2Adam/hiddenLayer3/bias/v
):'d
2Adam/outputLayer/kernel/v
#:!
2Adam/outputLayer/bias/v
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_39959
E__inference_sequential_layer_call_and_return_conditional_losses_39668
E__inference_sequential_layer_call_and_return_conditional_losses_39633
E__inference_sequential_layer_call_and_return_conditional_losses_39909?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_39988
*__inference_sequential_layer_call_fn_39733
*__inference_sequential_layer_call_fn_39797
*__inference_sequential_layer_call_fn_40017?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_39302?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+
inputLayer_input?????????
?2?
E__inference_inputLayer_layer_call_and_return_conditional_losses_40023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_inputLayer_layer_call_fn_40028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_40038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_hiddenLayer1_layer_call_fn_40047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40083
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40103?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_40116
3__inference_batch_normalization_layer_call_fn_40129?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_40140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_hiddenLayer2_layer_call_fn_40149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_40166
B__inference_dropout_layer_call_and_return_conditional_losses_40161?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_40171
'__inference_dropout_layer_call_fn_40176?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_40187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_hiddenLayer3_layer_call_fn_40196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_outputLayer_layer_call_and_return_conditional_losses_40207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_outputLayer_layer_call_fn_40216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
;B9
#__inference_signature_wrapper_39836inputLayer_input?
 __inference__wrapped_model_39302?!"+,12=?:
3?0
.?+
inputLayer_input?????????
? "9?6
4
outputLayer%?"
outputLayer?????????
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40083d4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_40103d4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
3__inference_batch_normalization_layer_call_fn_40116W4?1
*?'
!?
inputs??????????
p
? "????????????
3__inference_batch_normalization_layer_call_fn_40129W4?1
*?'
!?
inputs??????????
p 
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_40161^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_40166^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_40171Q4?1
*?'
!?
inputs??????????
p
? "???????????|
'__inference_dropout_layer_call_fn_40176Q4?1
*?'
!?
inputs??????????
p 
? "????????????
G__inference_hiddenLayer1_layer_call_and_return_conditional_losses_40038^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_hiddenLayer1_layer_call_fn_40047Q0?-
&?#
!?
inputs??????????
? "????????????
G__inference_hiddenLayer2_layer_call_and_return_conditional_losses_40140^!"0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_hiddenLayer2_layer_call_fn_40149Q!"0?-
&?#
!?
inputs??????????
? "????????????
G__inference_hiddenLayer3_layer_call_and_return_conditional_losses_40187]+,0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? ?
,__inference_hiddenLayer3_layer_call_fn_40196P+,0?-
&?#
!?
inputs??????????
? "??????????d?
E__inference_inputLayer_layer_call_and_return_conditional_losses_40023]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_inputLayer_layer_call_fn_40028P3?0
)?&
$?!
inputs?????????
? "????????????
F__inference_outputLayer_layer_call_and_return_conditional_losses_40207\12/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? ~
+__inference_outputLayer_layer_call_fn_40216O12/?,
%?"
 ?
inputs?????????d
? "??????????
?
E__inference_sequential_layer_call_and_return_conditional_losses_39633|!"+,12E?B
;?8
.?+
inputLayer_input?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39668|!"+,12E?B
;?8
.?+
inputLayer_input?????????
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39909r!"+,12;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_39959r!"+,12;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
*__inference_sequential_layer_call_fn_39733o!"+,12E?B
;?8
.?+
inputLayer_input?????????
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_39797o!"+,12E?B
;?8
.?+
inputLayer_input?????????
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_39988e!"+,12;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_40017e!"+,12;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
#__inference_signature_wrapper_39836?!"+,12Q?N
? 
G?D
B
inputLayer_input.?+
inputLayer_input?????????"9?6
4
outputLayer%?"
outputLayer?????????
