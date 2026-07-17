def csv_to_tsv_file(input_path="results.csv", output_path="results_tsv.csv"):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            # 쉼표(,)를 탭(\t)으로 바꿔서 새 파일에 기록
            tsv_line = line.strip().replace(",", "\t")
            outfile.write(tsv_line + "\n")

if __name__ == "__main__":
    csv_to_tsv_file("results.csv", "results_tsv.csv")