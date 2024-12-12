% SVM
clear; close all; clc;

%% Step 1: 데이터 로드 및 초기화
% 윈도우 바탕화면의 instu_data 경로 설정
desktop_path = fullfile(getenv('USERPROFILE'), 'Desktop');
base_folder = fullfile(desktop_path, 'instu_data');

% 악기 폴더와 라벨 설정
instrument_folders = {'gac', 'pia', 'sax', 'vio'};  % 각 폴더 이름
labels = {'gac', 'pia', 'sax', 'vio'};              % 각 악기의 라벨

% 데이터와 레이블 저장 변수 초기화
all_features = [];
all_labels = [];

%% Step 2: FIR 필터 적용 및 특징 추출
for i = 1:length(instrument_folders)
    folder_path = fullfile(base_folder, instrument_folders{i});
    audio_files = dir(fullfile(folder_path, '*.wav'));  % WAV 파일만 선택

    % 각 폴더에서 데이터를 처리
    local_features = [];
    local_labels = [];
    for j = 1:length(audio_files)
        file_path = fullfile(folder_path, audio_files(j).name);
        [audio, Fs] = audioread(file_path);

        % 오디오 데이터를 정규화
        audio = audio / max(abs(audio));

        % FIR 필터 적용 (평균 필터)
        M = 5;  % 필터 길이
        filtered_audio = filter(ones(1, M)/M, 1, audio);

        % 특징 추출 (MFCC, RMS, ZCR, Spectral Flatness, Crest Factor, Spectral Centroid)
        window_size = 1024;  % 윈도우 크기
        hop_size = 512;      % 윈도우 이동 크기
        analysis_window = hamming(window_size, 'periodic');  % Hamming 윈도우 생성
        mfcc_features = mfcc(filtered_audio, Fs, 'Window', analysis_window, ...
                              'OverlapLength', window_size - hop_size);
        rms_energy = sqrt(mean(filtered_audio .^ 2));  % RMS 에너지
        zcr = sum(abs(diff(sign(filtered_audio)))) / length(filtered_audio);  % ZCR
        spectral_flatness = geo_mean(abs(filtered_audio)) / mean(abs(filtered_audio));  % 스펙트럼 평탄도
        crest_factor = max(abs(filtered_audio)) / rms_energy;  % 크레스트 팩터
        spectral_centroid = sum((1:length(filtered_audio))' .* abs(filtered_audio)) / sum(abs(filtered_audio));  % 스펙트럼 중심
        spectral_bandwidth = sqrt(sum(((1:length(filtered_audio))' - spectral_centroid) .^ 2 .* abs(filtered_audio)) / sum(abs(filtered_audio)));  % 스펙트럼 대역폭

        % 모든 특징 병합
        feature_vector = [mean(mfcc_features, 1), rms_energy, zcr, spectral_flatness, crest_factor, ...
                          spectral_centroid, spectral_bandwidth];
        local_features = [local_features; feature_vector];
        local_labels = [local_labels; labels(i)];
    end

    % 각 클래스의 데이터를 통합
    all_features = [all_features; local_features];
    all_labels = [all_labels; local_labels];
end
disp('FIR 필터 적용 및 특징 추출 완료.');

%% Step 3: 데이터 정규화 및 분리
% 데이터를 [0, 1] 범위로 정규화
feature_min = min(all_features, [], 1);
feature_max = max(all_features, [], 1);
scaled_features = (all_features - feature_min) ./ (feature_max - feature_min);

% 데이터를 랜덤하게 섞기
num_samples = size(scaled_features, 1);
random_idx = randperm(num_samples);

% 훈련/테스트 데이터 비율 설정
train_ratio = 0.8;
train_idx = random_idx(1:round(train_ratio * num_samples));
test_idx = random_idx(round(train_ratio * num_samples) + 1:end);

% 데이터 분리
train_features = scaled_features(train_idx, :);
test_features = scaled_features(test_idx, :);

% 레이블 데이터를 categorical 형식으로 변환
train_labels = categorical(all_labels(train_idx));
test_labels = categorical(all_labels(test_idx));

disp('데이터 정규화 및 분리 완료.');

%% Step 4: SVM 모델 학습
% 클래스 가중치 계산
class_weights = 1 ./ countcats(train_labels);
weights = arrayfun(@(x) class_weights(find(categories(train_labels) == x, 1)), train_labels);

% 하이퍼파라미터 최적화
template = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);

% 하이퍼파라미터 최적화 설정
mdl = fitcecoc(train_features, train_labels, ...
    'Learners', template, ...
    'Weights', weights, ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', false, ...  % 최적화 과정을 시각화하려면 true로 설정
        'Verbose', 1));

disp('SVM 모델 학습 및 최적화 완료.');

%% Step 5: 모델 평가
% 테스트 데이터로 예측
predicted_labels = predict(mdl, test_features);

% 정확도 계산
accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;
disp(['모델 정확도: ', num2str(accuracy), '%']);

% Confusion Matrix 시각화
confusionchart(test_labels, predicted_labels);
title('Confusion Matrix - SVM Model');

% 악기별 정확도 계산
categories_list = categories(test_labels);
for i = 1:length(categories_list)
    category = categories_list{i};
    idx = test_labels == category;
    category_accuracy = sum(predicted_labels(idx) == test_labels(idx)) / sum(idx) * 100;
    disp(['악기: ', category, ' 정확도: ', num2str(category_accuracy), '%']);
end

%%
% clear, close all, clc;
% 
% %% Step 1: 데이터 로드 및 초기화
% % 윈도우 바탕화면의 instu_data 경로 설정
% desktop_path = fullfile(getenv('USERPROFILE'), 'Desktop');
% base_folder = fullfile(desktop_path, 'instu_data');
% 
% % 악기 폴더와 라벨 설정
% instrument_folders = {'gac', 'pia', 'sax', 'vio'};  % 각 폴더 이름
% labels = {'gac', 'pia', 'sax', 'vio'};              % 각 악기의 라벨
% 
% % 데이터와 레이블 저장 변수 초기화
% all_features = [];
% all_labels = [];
% 
% % 병렬 풀 시작 (필요 시)
% if isempty(gcp('nocreate'))
%     parpool;  % 병렬 풀 시작
% end
% 
% %% Step 2: FIR 필터 적용 및 특징 추출
% parfor i = 1:length(instrument_folders)  % 병렬 처리
%     folder_path = fullfile(base_folder, instrument_folders{i});
%     audio_files = dir(fullfile(folder_path, '*.wav'));  % WAV 파일만 선택
% 
%     % 각 폴더에서 데이터를 처리
%     local_features = [];
%     local_labels = [];
%     for j = 1:length(audio_files)
%         file_path = fullfile(folder_path, audio_files(j).name);
%         [audio, Fs] = audioread(file_path);
% 
%         % % 오디오 데이터를 다운샘플링 (2배)
%         audio = resample(audio, 1, 2);
%         Fs = Fs / 2;
% 
%         % 오디오 데이터를 정규화
%         audio = audio / max(abs(audio));
% 
%         % FIR 필터 적용 (평균 필터)
%         M = 15;  % 필터 길이
%         filtered_audio = filter(ones(1, M)/M, 1, audio);
% 
%         % 특징 추출 (MFCC 및 RMS 에너지)
%         window_size = 1024;  % 특징 추출에 사용할 윈도우 크기
%         hop_size = 512;      % 윈도우 간 이동 크기
%         analysis_window = hamming(window_size, 'periodic');  % Hamming 윈도우 생성
%         mfcc_features = mfcc(filtered_audio, Fs, 'Window', analysis_window, ...
%                               'OverlapLength', window_size - hop_size);
%         rms_energy = sqrt(mean(filtered_audio .^ 2));  % RMS 에너지 계산
% 
%         % 특징 병합
%         feature_vector = [mean(mfcc_features, 1), rms_energy];
%         local_features = [local_features; feature_vector];  % 특징 추가
%         local_labels = [local_labels; labels(i)];           % 라벨 추가
%     end
% 
%     % 병렬 처리 결과 통합
%     all_features = [all_features; local_features];
%     all_labels = [all_labels; local_labels];
% end
% disp('FIR 필터 적용 및 특징 추출 완료.');
% 
% %% Step 3: 데이터 분리 (훈련 데이터와 테스트 데이터)
% % 데이터를 랜덤하게 섞기
% num_samples = size(all_features, 1);
% random_idx = randperm(num_samples);
% 
% train_ratio = 0.8;  % 훈련 데이터 비율
% train_idx = random_idx(1:round(train_ratio * num_samples));
% test_idx = random_idx(round(train_ratio * num_samples) + 1:end);
% 
% train_features = all_features(train_idx, :);
% test_features = all_features(test_idx, :);
% 
% % 레이블 데이터를 categorical 형식으로 변환
% train_labels = categorical(all_labels(train_idx));
% test_labels = categorical(all_labels(test_idx));
% 
% disp('데이터 분리 완료.');
% 
% %% Step 3-1: 특징 스케일링 (정규화)
% % 모든 특징을 [0, 1] 범위로 정규화
% feature_min = min(all_features, [], 1);
% feature_max = max(all_features, [], 1);
% scaled_features = (all_features - feature_min) ./ (feature_max - feature_min);
% 
% % 데이터 분리
% num_samples = size(scaled_features, 1);
% random_idx = randperm(num_samples);
% 
% train_ratio = 0.8;  % 훈련 데이터 비율
% train_idx = random_idx(1:round(train_ratio * num_samples));
% test_idx = random_idx(round(train_ratio * num_samples) + 1:end);
% 
% train_features = scaled_features(train_idx, :);
% test_features = scaled_features(test_idx, :);
% 
% % 레이블 데이터를 categorical 형식으로 변환
% train_labels = categorical(all_labels(train_idx));
% test_labels = categorical(all_labels(test_idx));
% 
% disp('데이터 스케일링 및 분리 완료.');
% 
% %% Step 4: Random Forest 모델 학습 (하이퍼파라미터 튜닝)
% num_trees = 300;  % 트리 개수 증가
% min_leaf_size = 3;  % 리프 노드 최소 크기 설정
% max_num_splits = 50;  % 최대 분할 수 제한
% 
% mdl = TreeBagger(num_trees, train_features, train_labels, ...
%                  'OOBPrediction', 'On', 'Method', 'classification', ...
%                  'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
% disp('Random Forest 모델 학습 완료.');
% 
% 
% %% Step 5: 모델 평가
% % 테스트 데이터를 사용해 예측
% [predicted_labels, scores] = predict(mdl, test_features);
% predicted_labels = categorical(predicted_labels);
% 
% % 정확도 계산
% accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;
% disp(['모델 정확도: ', num2str(accuracy), '%']);
% 
% % Confusion Matrix 시각화
% confusionchart(test_labels, predicted_labels);
% title('Confusion Matrix - Tuned Random Forest');
% 
% % 악기별 정확도 계산
% categories_list = categories(test_labels);
% for i = 1:length(categories_list)
%     category = categories_list{i};
%     idx = test_labels == category;
%     category_accuracy = sum(predicted_labels(idx) == test_labels(idx)) / sum(idx) * 100;
%     disp(['악기: ', category, ' 정확도: ', num2str(category_accuracy), '%']);
% end
% 
% % Out-of-Bag Error 시각화
% figure;
% oobErrorBaggedEnsemble = oobError(mdl);
% plot(oobErrorBaggedEnsemble);
% xlabel('Number of Trees');
% ylabel('Out-of-Bag Classification Error');
% title('OOB Error vs Number of Trees');