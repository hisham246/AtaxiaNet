for i = 1:23
    filename = sprintf('Limb Lengths/Test/test_limb_lengths_%s.csv', string(i));
    L = csvread(filename,1,1); % Limb lengths data matrix
    S = zeros(31, 10); % Augmented matrix
    M = [];
    for k = 1:size(L,1)
        count = 1;
        V = L(k,:); % each row of mat
        for q = 1:1:5
            comb = nchoosek(1:1:5, q);
            for b = 1:1:size(comb, 1)
                S(count, :) = V;
                temp1 = V(comb(b, :));
                temp2 = V(comb(b, :) + 5);
                S(count, comb(b, :)) = temp2;
                S(count, comb(b, :) + 5) = temp1;
                count = count + 1;
            end
        end
        aug = [V;S];
        M = vertcat(M, aug);
        S = zeros(31, 10);
    end
    T = array2table(M);
    T.Properties.VariableNames(1:10) = {'LeftUpperArm','LeftLowerArm'...
        'LeftTorso', 'LeftUpperLeg', 'LeftLowerLeg', 'RightUpperArm'...
        'RightLowerArm', 'RightTorso', 'RightUpperLeg', 'RightLowerLeg'};
    writetable(T, sprintf('Data Augmentation/Augmented Datasets/Test/test_augmented_%s.csv', string(i)))
end
