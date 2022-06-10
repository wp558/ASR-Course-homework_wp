 **语音识别课程作业  Python、c++**

02-feature-extraction:语音信号特征提取 包括MFCC Fbank特征   

03-GMM-EM:实现0-9数字识别

04-hmm：以盒子摸球为例，实现前向后向概率计算、viterbi解码等

05-GMM-HMM：基于GMM-HMM的0-9语音识别系统

06-DNN-HMM：基于DNN-HMM的0-9语音识别系统

07-lm：   

          完成N-gram计数和Witten-Bell算法的编写   

          基于THCHS30中文语料，采用modified Knser-Ney算法训练相应的语言模型并计算困惑度
          
08-wfst： 

          对两个fst进行生成、合并、输出等操作

          运行kaldi/egs/mini_librispeech至少训练完3音素模型tri1,生成G.fst.
          
          用 tri1 模型和 tgsmall 构建的 HCLG 图解码 dev_clean_2 集合的“1272-135031-0009”句，输
          
          出 Lattice 和 CompactLattice 的文本格式。
          
          使用中生成的 tglarge 的 G.fst 和 steps/lmrescore.sh 对
          
          exp/tri1/decode_nosp_tgsmall_dev_clean_2 中的 lattice 重打分，汇报 wer.
