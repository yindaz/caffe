% file_PN = '/home/yindazhang/Desktop/caffe_ionic/models/mix_noise/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-224410.9094';
% file_PP = '/home/yindazhang/Desktop/caffe_ionic/models/mix_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-222926.3892';

% file_PN = '/home/yindazhang/Desktop/caffe_ionic/models/positive_noise/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-111425.13481';
% file_PP = '/home/yindazhang/Desktop/caffe_ionic/models/positive_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-111420.13473';

file_PN = '/home/yindazhang/Desktop/caffe_ionic/models/mix_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-222926.3892';
file_PP = '/home/yindazhang/Desktop/caffe_ionic/models/positive_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141031-111420.13473';


file_PN_temp = './script/PN.txt';
file_PP_temp = './script/PP.txt';
system(sprintf('cp %s %s', file_PN, file_PN_temp));
system(sprintf('cp %s %s', file_PP, file_PP_temp));

% FID = fopen(file_PN_temp,'r');
string_PN = file2string(file_PN_temp);
% fclose(FID);
% FID = fopen(file_PP_temp,'r');
string_PP = file2string(file_PP_temp);
% fclose(FID);

testPattern = 'Test net output #0: accuracy = ';
testOffset = length(testPattern);

PN_record = strfind(string_PN, testPattern);
PP_record = strfind(string_PP, testPattern);

PN_loss = zeros(1000,1);
PP_loss = zeros(1000,1);
for i = 1:length(PN_record)
    offset = PN_record(i) + testOffset;
    PN_loss(i) = str2num(string_PN(offset:offset+5));
end
for i = 1:length(PP_record)
    offset = PP_record(i) + testOffset;
    PP_loss(i) = str2num(string_PP(offset:offset+5));
end

figure(1); clf;
plot(1:length(1:2:length(PN_record)), PN_loss(1:2:length(PN_record)), 'r');
hold on;
plot(1:length(PP_record), PP_loss(1:length(PP_record)), 'c');

legend('Positive Noise', 'Positive Pure');

% filename = './models/positive_noise/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141026-143428.15242';
% % filename = './models/positive_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141026-143525.16907';
% % FID = fopen(filename,'r');
% S = file2string(filename);
% 
% trainPattern = 'Train net output #0: loss = ';
% testPattern = 'Test net output #0: accuracy = ';
% 
% trainrecord = strfind(S, trainPattern);
% testrecord = strfind(S, testPattern);
% 
% trainLoss = zeros(length(trainrecord),1);
% for i = 1:length(trainrecord)
%     offset = trainrecord(i)+length(trainPattern);
%     trainLoss(i) = str2num(S(offset:offset+5));
% end
% 
% testLoss = zeros(length(testrecord),1);
% for i = 1:length(testrecord)
%     offset = testrecord(i)+length(testPattern);
%     testLoss(i) = str2num(S(offset:offset+5));
% end