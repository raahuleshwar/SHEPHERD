INPUT=project_config.PROJECT_DIR/patients/mygene2_patients/genes.csv

[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r gene 
do
    echo "$gene"
    wget https://www.mygene2.org/MyGene2/api/data/export/$gene
done < $INPUT
