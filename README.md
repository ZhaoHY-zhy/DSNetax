# DSNetax
This is a deep learning method for large-scale bacterial species level classification identification. You can use it according to the following rules of use.
Paper: https://doi.org/10.1093/bib/bbae157
![The experimental process of this method is as follows:](https://github.com/ZhaoHY-zhy/pics/blob/main/Figure%201.png)

**Method 1:**  

   ```
    python merge_annotation.py --i_fasta test.fasta --o_seq2num test --o_taxo_class test.txt  
   ```
    
    The required parameters：  
     --i_fasta: The fasta file path to be annotated.  
     --o_seq2num: Processed file path without file suffix.  
     --o_taxo_class: The path of results annotation file (.txt).  
     
    The optional parameters:  
     --i_dnabert_m: The dnabert model file path. The default value is k_3-108_dnabert.  
     --i_DSNetax_m: The DSNetax model file path. The default value is trained_DSNetax.pth.  
     --i_result_nums: Number of output results. The default value is 1. The optional value is 1 or 5.  
     --i_seqchunk: How many sequences are processed each time. The default value is 1000.  
     
**Method 2:**  

   **This method takes two steps to complete.**

   **The first step:**
   ```
    python .\seq2tensor_k3.py --i_fasta test.fasta --o_seq2num test  
   ```
    
    The required parameters：  
     --i_fasta: The fasta file path to be annotated.  
     --o_seq2num: Processed file path without file suffix.   
     
    The optional parameters:  
     --i_dnabert_m: The dnabert model file path. The default value is k_3-108_dnabert.  
     --i_seqchunk: How many sequences are processed each time. The default value is 1000.  
     
   **The second step:**
   ```
   python .\DSNetax_result.py --i_seq2num test --o_taxo_class test.txt  
   ```
   
    The required parameters：   
     --o_seq2num: Processed file path without file suffix.  
     --o_taxo_class: The path of results annotation file (.txt).  
     
    The optional parameters:   
     --i_DSNetax_m: The DSNetax model file path. The default value is trained_DSNetax.pth.  
     --i_result_nums: Number of output results. The default value is 1. The optional value is 1 or 5.  

The DNABERT model and the trained DSNetax model have been stored in Baidu web disk, and the specific address and extraction code are as follows:
