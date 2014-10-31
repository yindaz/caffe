filename = './models/positive_noise/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141026-143428.15242';
% filename = './models/positive_pure/log/caffe.pvg-gpu-desktop.yindazhang.log.INFO.20141026-143525.16907';
% FID = fopen(filename,'r');
S = file2string(filename);

trainPattern = 'Train net output #0: loss = ';
testPattern = 'Test net output #0: accuracy = ';

trainrecord = strfind(S, trainPattern);
testrecord = strfind(S, testPattern);

trainLoss = zeros(length(trainrecord),1);
for i = 1:length(trainrecord)
    offset = trainrecord(i)+length(trainPattern);
    trainLoss(i) = str2num(S(offset:offset+5));
end

testLoss = zeros(length(testrecord),1);
for i = 1:length(testrecord)
    offset = testrecord(i)+length(testPattern);
    testLoss(i) = str2num(S(offset:offset+5));
end