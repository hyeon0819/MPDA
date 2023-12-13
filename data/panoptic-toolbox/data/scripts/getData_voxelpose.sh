sequences=(
    '160422_ultimatum1'
    '160906_ian1'
    '160224_haggling1'
    '160906_band1'
    '160226_haggling1'
    '160906_ian2'
    '160906_band2'
    '161202_haggling1'
    '160906_ian3'
    '160906_band3'

    '170404_haggling_a1'
    '170915_toddler5'
    '161029_build1' 
    '170221_haggling_b1'

    '160906_pizza1'
    '160422_haggling1'
    '160906_ian5'
    '160906_band4'
)

for seq in "${sequences[@]}"
do
    echo ${seq}
    cmd=$(bash scripts/getData_5.sh ${seq} 0 5)
    cmd=$(bash scripts/extractAll.sh ${seq})
    echo $cmd
    # eval $cmd
done

# for seq in "${sequences[@]}"
# do
#     echo ${seq}
#     cmd=$(bash scripts/extractAll.sh ${seq})
#     echo $cmd
#     # eval $cmd
# done