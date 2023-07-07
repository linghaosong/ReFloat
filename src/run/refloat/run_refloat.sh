EXE_PATH=../../
RUN=${EXE_PATH}/cgrefloat

# matrix directory
DIR=../../../matrices

# set the RUN to CG or BICG
# CG solver
SOLVER=0
# BiCG solver
#SOLVER=1

MAX_ITE=1000

${RUN} ${DIR}/crystm01/crystm01.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/minsurfo/minsurfo.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/crystm02/crystm02.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/shallow_water1/shallow_water1.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/wathen100/wathen100.mtx ${MAX_ITE} ${SOLVER} 3 16
${RUN} ${DIR}/gridgena/gridgena.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/wathen120/wathen120.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/crystm03/crystm03.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/thermomech_TC/thermomech_TC.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/Dubcova2/Dubcova2.mtx ${MAX_ITE} ${SOLVER} 3 16
${RUN} ${DIR}/thermomech_dM/thermomech_dM.mtx ${MAX_ITE} ${SOLVER} 3 8
${RUN} ${DIR}/qa8fm/qa8fm.mtx ${MAX_ITE} ${SOLVER} 3 8
