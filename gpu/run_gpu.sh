# set to not print residual
PRINT_RES=
# set to print residual
#PRINT_RES=1

# set the RUN to CG or BICG
RUN=gpucg
#RUN=gpubicg

# matrix directory
DIR=../matrices

MAX_ITE=1000

./${RUN} ${DIR}/crystm01/crystm01.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/minsurfo/minsurfo.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/crystm02/crystm02.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/shallow_water1/shallow_water1.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/wathen100/wathen100.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/gridgena/gridgena.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/wathen120/wathen120.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/crystm03/crystm03.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/thermomech_TC/thermomech_TC.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/Dubcova2/Dubcova2.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/thermomech_dM/thermomech_dM.mtx ${MAX_ITE} ${PRINT_RES}
./${RUN} ${DIR}/qa8fm/qa8fm.mtx ${MAX_ITE} ${PRINT_RES}
