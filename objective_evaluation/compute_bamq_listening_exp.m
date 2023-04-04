addpath("bam-q/BAMQ")
rv = [];
sc = [];
m = [];
score = [];
for reverb = ["anech", "medrev", "strongrev"]
    for scenario = ["pink_noise", "drums+saw", "string_quartet", ...
                    "two_speakers", "speech+noise"]
                
        ref_filepath = "../generate_stimuli/rendered_audio/" + scenario ...
                       + "_" + reverb + "_" + "ref_binau.wav";
        [xref, fs] = audioread(ref_filepath);
        for method = ["mono", "foa", "param1", "param2", ...
                      "foa_amb_param1", "foa_amb_param2", "harpex"]
            if method == "mono"
                filepath = "../generate_stimuli/rendered_audio/" + ...
                           scenario + "_" + reverb + "_mono.wav";
            else
                filepath = "../generate_stimuli/rendered_audio/" + ...
                           scenario + "_" + reverb + "_" + method + ...
                           "_binau.wav";
            end
            [xpred, fs] = audioread(filepath);
            
            [qual, ~, ~, ~] = BAMQ4Public_restruct(xref, xpred, fs);
            rv = [rv; reverb];
            sc = [sc; scenario];
            m = [m; method];
            score = [score; qual];
        end
    end
end
scoretable = table(rv, sc, m, score, 'VariableNames', {'reverberation', ...
                   'scenario', 'method', 'score'});
writetable(scoretable, 'bamq.csv');
