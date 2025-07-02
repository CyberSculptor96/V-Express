step=113400

ffmpeg \
-i "exp_output-20h/stage_2-0630_0634/validation/video-${step}.mp4" \
-i "exp_output-40h/stage_2-0630_0636/validation/video-${step}.mp4" \
-i "exp_output-80h/stage_2-0630_0640/validation/video-${step}.mp4" \
-i "exp_output-160h/stage_2-0630_0636/validation/video-${step}.mp4" \
-filter_complex \
"[0:v]drawtext=text='20h':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=50[v0]; \
 [1:v]drawtext=text='40h':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=50[v1]; \
 [2:v]drawtext=text='80h':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=50[v2]; \
 [3:v]drawtext=text='160h':fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=50[v3]; \
 [v0][v1][v2][v3]hstack=inputs=4[v]; \
 [0:a]anull[a]" \
-map "[v]" \
-map "[a]" \
-c:v libx264 -preset fast -crf 23 \
-c:a aac -b:a 192k \
-pix_fmt yuv420p \
-y user/demo/stage2-step_${step}.mp4