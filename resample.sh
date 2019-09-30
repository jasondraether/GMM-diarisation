mkdir outdir
for i in labeled_wavs/Ryan/*.wav; do
	ffmpeg -i $i -ar 16000 outdir/$i;
done
