clc;
% 데이터 로드 및 초기화
desktop_path = fullfile(getenv('USERPROFILE'), 'Desktop');
base_folder = fullfile(desktop_path, 'instu_data'); % 상위 폴더
output_folder = fullfile(desktop_path, 'no_filtered_data'); % 결과 저장 폴더

% 출력 폴더 생성
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 악기별 폴더 이름 설정
instrumentFolders = struct( ...
    'violin', 'vio', ...
    'saxophone', 'sax', ...
    'guitar', 'gac', ...
    'piano', 'pia' ...
);

% FIR 필터 차수
filterOrder = 50;

% 악기별 필터 대역 설정
filterBands = struct( ...
    'violin', [196 3000], ... % 바이올린 대역 (예시)
    'saxophone', [110 2500], ... % 색소폰 대역 (예시)
    'guitar', [80 5000], ... % 어쿠스틱 기타 대역 (예시)
    'piano', [27 8000] ... % 피아노 대역 (예시)
);

% 데이터와 레이블 저장 변수 초기화
  all_features = []; % 특징 데이터를 저장할 배열 초기화
  all_labels = [];   % 레이블 데이터를 저장할 배열 초기화


% 악기 폴더를 순회하며 처리
fieldNames = fieldnames(instrumentFolders);

for i = 1:length(fieldNames)
    instrument = fieldNames{i}; % 악기 이름
    folderName = instrumentFolders.(instrument); % 폴더 이름
    instrumentFolderPath = fullfile(base_folder, folderName); % 악기 데이터 경로
    
    
    %% 수정된 코드
    % (수정전) 해당 폴더 내 WAV 파일 검색 (파일 이름에 해당 악기 접두사 포함)
    % switch instrument
    %     case 'violin'
    %         wavFiles = dir(fullfile(instrumentFolderPath, '[vio]*.wav')); % [vio]로 시작하는 파일
    %     case 'saxophone'
    %         wavFiles = dir(fullfile(instrumentFolderPath, '[sax]*.wav')); % [sax]로 시작하는 파일
    %     case 'guitar'
    %         wavFiles = dir(fullfile(instrumentFolderPath, '[gac]*.wav')); % [gac]로 시작하는 파일
    %     case 'piano'
    %         wavFiles = dir(fullfile(instrumentFolderPath, '[pia]*.wav')); % [pia]로 시작하는 파일
    % end

    % (수정후) WAV 파일 검색 (모든 WAV 파일 검색)
    wavFiles = dir(fullfile(instrumentFolderPath, '*.wav'));
    
    % 파일 존재 여부 확인
    if isempty(wavFiles)
        fprintf('폴더 "%s"에 WAV 파일이 없습니다.\n', instrumentFolderPath);
        continue; % 다음 폴더로 이동
    end

    %%
  
    % 출력 폴더 생성
    outputFolderPath = fullfile(output_folder, folderName);
    if ~exist(outputFolderPath, 'dir')
        mkdir(outputFolderPath);
    end
    
    % 폴더 내 파일 처리
    for k = 1:length(wavFiles)
        fileName = wavFiles(k).name;
        fullFilePath = fullfile(instrumentFolderPath, fileName);
        
        % WAV 파일 읽기
        [audioData, fs] = audioread(fullFilePath);
        
        % 필터 설계 및 적용
        cutoffFrequencies = filterBands.(instrument) / (fs / 2); % 정규화
        b = fir1(filterOrder, cutoffFrequencies, 'bandpass'); % FIR 필터 설계

        %% 수정된 코드
        % (수정전) filteredData = filter(b, 1, audioData); 

        % (수정후) 각 채널별 필터링
        filteredData = zeros(size(audioData));
        for ch = 1:size(audioData, 2)
            filteredData(:, ch) = filter(b, 1, audioData(:, ch));
        end
        %%
        
        % 결과 저장
        outputFileName = fullfile(outputFolderPath, [fileName(1:end-4) '_filtered.wav']);
        audiowrite(outputFileName, filteredData, fs);
        
        % 진행 상황 출력
        fprintf('Processed and saved: %s\n', outputFileName);

        %% 추가된 코드
        % 특징 추출 (MFCC, RMS 에너지 등)
        mfcc_features_ch1 = mfcc(filteredData(:, 1), fs, 'NumCoeffs', 13);
        rms_energy_ch1 = sqrt(mean(filteredData(:, 1) .^ 2));
        
        if size(audioData, 2) > 1 % 두 번째 채널이 있는 경우
            mfcc_features_ch2 = mfcc(filteredData(:, 2), fs, 'NumCoeffs', 13);
            rms_energy_ch2 = sqrt(mean(filteredData(:, 2) .^ 2));
        else
            mfcc_features_ch2 = [];
            rms_energy_ch2 = [];
        end
        
        % 평균 MFCC를 계산하고 병합
        avg_mfcc_ch1 = mean(mfcc_features_ch1, 1);
        if ~isempty(mfcc_features_ch2)
            avg_mfcc_ch2 = mean(mfcc_features_ch2, 1);
            feature_vector = [avg_mfcc_ch1, rms_energy_ch1, avg_mfcc_ch2, rms_energy_ch2];
        else
            feature_vector = [avg_mfcc_ch1, rms_energy_ch1];
        end
        
        all_features = [all_features; feature_vector];
        all_labels = [all_labels; {instrument}];

    end
end

disp('모든 폴더 및 파일 필터링 완료!');


%% 데이터 전처리
% all_labels 데이터 형식 확인 및 변환
disp('all_labels의 데이터 형식:');
disp(class(all_labels));

if isnumeric(all_labels) % 숫자형 배열인 경우
    all_labels = categorical(all_labels);
elseif iscell(all_labels) % 셀 배열인 경우
    valid_idx = ~cellfun('isempty', all_labels) & ~strcmp(all_labels, '');
    all_features = all_features(valid_idx, :);
    all_labels = all_labels(valid_idx);
elseif isstring(all_labels) || ischar(all_labels) % 문자열 배열 또는 문자형 배열인 경우
    valid_idx = all_labels ~= "";
    all_features = all_features(valid_idx, :);
    all_labels = all_labels(valid_idx);
else
    error('all_labels는 셀 배열, 문자열 배열, 문자형 배열, 숫자형 배열 중 하나여야 합니다.');
end

% 고유 클래스 이름 출력
disp('수정된 고유 클래스 이름:');
disp(categories(categorical(all_labels)));

% 데이터 랜덤 섞기
num_samples = size(all_features, 1);
random_idx = randperm(num_samples);

% 훈련/테스트 데이터 분리 비율
train_ratio = 0.8;
train_idx = random_idx(1:round(train_ratio * num_samples));
test_idx = random_idx(round(train_ratio * num_samples) + 1:end);

% 데이터 분리
train_features = all_features(train_idx, :);
test_features = all_features(test_idx, :);

% 범주형 라벨 변환
train_labels = categorical(all_labels(train_idx));
test_labels = categorical(all_labels(test_idx));

% 고유 클래스 이름 확인
disp('훈련 데이터 클래스:');
disp(categories(train_labels));
disp('테스트 데이터 클래스:');
disp(categories(test_labels));

%% Random Forest 모델 학습 (앙상블 알고리즘)
num_trees = 300; % 트리 개수 설정
min_leaf_size = 3; % 리프 노드 최소 크기
max_num_splits = 50; % 최대 분할 수

mdl = TreeBagger(num_trees, train_features, train_labels, ...
                 'OOBPrediction', 'On', 'Method', 'classification', ...
                 'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
disp('Random Forest 모델 학습 완료.');

%% 모델 평가
% 테스트 데이터로 예측
[predicted_labels, scores] = predict(mdl, test_features);
predicted_labels = categorical(predicted_labels);

% 정확도 계산
accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;
disp(['모델 정확도: ', num2str(accuracy), '%']);

% Confusion Matrix 시각화
confusionchart(test_labels, predicted_labels);
title('Confusion Matrix - Random Forest');

% 악기별 정확도 계산
categories_list = categories(test_labels);
for i = 1:length(categories_list)
    category = categories_list{i};
    idx = test_labels == category;
    category_accuracy = sum(predicted_labels(idx) == test_labels(idx)) / sum(idx) * 100;
    disp(['악기: ', category, ' 정확도: ', num2str(category_accuracy), '%']);
end

% OOB Error 시각화
figure;
oobErrorBaggedEnsemble = oobError(mdl);
plot(oobErrorBaggedEnsemble);
xlabel('Number of Trees');
ylabel('Out-of-Bag Classification Error');
title('OOB Error vs Number of Trees');
