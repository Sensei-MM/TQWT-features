clc;clear
fs = 200;
band_num = 8;
phase_len = 1024;
Q = 1; r = 3; J = 7;
path = 'D:\Dataset\SEED\Preprocessed_EEG\';
save_path = 'H:\SEED-tqwt2-8band-q1r3j7\';
file = dir(fullfile(path,'*.mat'));
subnames = {file.name}';
subnames = subnames(1:length(subnames)-1);

for sub = 1:length(subnames)
    sub_tqwt = struct();
    sub_name = char(subnames(sub));
    save_sub_name = sub_name(1:length(sub_name)-4);
    sub_eeg = load(strcat(path,sub_name));
    vid_names = fieldnames(sub_eeg);
    for vid = 1:length(vid_names)
        fprintf('Processing subject: %d ,vid: %d\n',sub,vid);
        vid_name = char(vid_names(vid));
        vid_eeg = sub_eeg.(vid_name);
        len_vid_eeg = size(vid_eeg,2);
        phase = floor(len_vid_eeg/phase_len);
        vid_band_tqwt = zeros(size(vid_eeg,1),phase*phase_len,band_num);
        for channel = 1:size(vid_eeg,1)
           temp = [];
           for i = 1:phase
              
              x = vid_eeg(channel , ((phase-1) * phase_len+1):phase*phase_len); 
              N = size(x,2);
              w = tqwt_radix2(x, Q, r, J);
%               b = {2:3 , 4 , 5 ,6,7};  %tqwt1 tqwt subbands from which to reconstruct
              b = {1,2,3,4,5,6,7,8};%tqwt2
              y = tqwt_bands(b, w, Q, r, N);
              temp = [temp y];
           end
           vid_band_tqwt(channel,:,:) = temp';    
        end
        sub_tqwt.(vid_name) = vid_band_tqwt;
    end
    %%
    save_path_name = strcat(save_path,sub_name);
    save(char(save_path_name),'sub_tqwt');
end