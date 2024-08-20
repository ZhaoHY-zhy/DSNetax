# DSNetax
This is a deep learning method for large-scale bacterial species level classification identification. You can use it according to the following rules of use.
Paper: https://doi.org/10.1093/bib/bbae157
![The experimental process of this method is as follows:](https://github.com/ZhaoHY-zhy/pics/blob/main/Figure%201.png)

**Method 1:**

   python merge_annotation.py --i_fasta test.fasta --o_seq2num test --o_taxo_class test.txt
   The required parameters：
     --i_fasta: The fasta file path to be annotated. 
     --o_seq2num: Processed file path without file suffix.
     --o_taxo_class: The path of results annotation file (.txt).
    The optional parameters:
      -i_dnabert_m", type=str, help="The dnabert model file path. ", default='k_3-108_dnabert', required=False)
    parser.add_argument("--i_DSNetax_m", type=str, help="The DSNetax model file path. ", default='trained_DSNetax.pth', required=False)
    parser.add_argument("--i_result_nums", type=str, help="Number of output results. ", default=1, choices=['1', '5'], required=False)
    parser.add_argument("--i_seqchunk", type=str, help="How many sequences are processed each time.", default=1000, required=False)
