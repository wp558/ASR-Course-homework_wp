for test in dev_clean_2; do
  steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_tgsmall data/lang_nosp_test_tglarge \
  data/$test exp/tri1/decode_nosp_tgsmall_$test
done