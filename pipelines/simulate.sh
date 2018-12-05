#!/usr/bin/env sh

# select steps of the pipeline
STEPS=$1

# default is all steps
if [ -z "$STEPS" ]
then
	STEPS="1,2,3,4"
fi

MASTER_PATH=/Users/ruthiejohnson/Development/g_coloc
SCRIPT_DIR=${MASTER_PATH}/scripts
SRC_DIR=${MASTER_PATH}/src
DATA_DIR=${MASTER_PATH}/data
RESULTS_DIR=${MASTER_PATH}/results

SIM_NAME=test
P_SIM=".94,.02,.02,.02"
H1=0.10
H2=0.02
RHO_G=0
RHO_E=0
N1=1000
N2=100000
Ns=0
M=100
LD_FILE=${DATA_DIR}/identity.${M}.ld

SEED=2018 # can replace with SGE_TASK_ID
ITS=500

DATE=`date '+%Y-%m-%d %H:%M:%S'`
echo $DATE" Starting simulation for g-coloc: "${SIM_NAME}

# STEP 1: Simulate loci
if [[ "$STEPS" =~ "1" ]]
then
  DATE=`date '+%Y-%m-%d %H:%M:%S'`
	echo $DATE" Simulting loci"
  python ${SCRIPT_DIR}/simulate.py \
    --name $SIM_NAME \
    --p_sim $P_SIM \
    --h1_sim $H1 \
    --h2_sim $H2 \
    --rhoG_sim $RHO_G \
    --rhoE_sim $RHO_E \
    --N1 $N1 \
    --N2 $N2 \
    --Ns $Ns \
    --M $M \
    --ld_file $LD_FILE \
    --seed $SEED \
    --outdir $DATA_DIR
fi


# STEP 2: Transform effect sizes
GWAS_FILE=${DATA_DIR}/${SIM_NAME}.${SEED}.txt
if [[ "$STEPS" =~ "2" ]]
then
	DATE=`date '+%Y-%m-%d %H:%M:%S'`
	echo $DATE" Transforming effect sizes"
	python ${SCRIPT_DIR}/transform_loci.py --gwas_file $GWAS_FILE --ld_file $LD_FILE
fi

# STEP 3: take 1/2 power of LD
if [[ "$STEPS" =~ "3" ]]
then
	python ${SCRIPT_DIR}/half_ld.py --ld_file $LD_FILE
fi

# STEP 4: Colcalization
LD_HALF_FILE=${LD_FILE%.*}.half_ld
if [[ "$STEPS" =~ "4" ]]
then
  python ${SRC_DIR}/g_coloc.py \
  --name $SIM_NAME \
  --gwas_file $GWAS_FILE \
  --ld_half_file $LD_HALF_FILE \
  --seed $SEED \
  --outdir $RESULTS_DIR \
  --h1 $H1 \
  --h2 $H2 \
  --rhoG $RHO_G \
  --rhoE $RHO_E \
  --N1 $N1 \
  --N2 $N2 \
  --Ns $Ns \
  --M $M \
  --its $ITS
fi
