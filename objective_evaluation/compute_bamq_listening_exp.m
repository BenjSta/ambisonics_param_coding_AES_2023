addpath("bam-q/BAMQ")
mystr = "";
quals = zeros(3, 5, 7);
for reverb = ["anech", "medrev", "strongrev"]
    disp(mystr)
    mystr = mystr + "\\midrule";
    mystr = mystr + sprintf("\\multirow{6}{*}{\\rotatebox[origin=c]{90}{\\textit{%s}}}", reverb);
    for scenario = ["pink_noise", "drums+saw", "string_quartet", "two_speakers", "speech+noise"]
        ref_filepath = "rendered_audio/" + scenario + "_" + reverb + "_" + "ref_binau.wav";
        [xref1, fs] = audioread(ref_filepath);
        mystr = mystr + "& " + sprintf("\\textit{%s}", strrep(scenario, '_', '\_'));
        for method = ["mono", "foa", "param1", "param2", "foa_amb_param1", "foa_amb_param2", "harpex"]
            if method == "mono"
                filepath = "rendered_audio/" + scenario + "_" + reverb + "_mono.wav";
            else
                filepath = "rendered_audio/" + scenario + "_" + reverb + "_" + method + "_binau.wav";
            end
            [xpred, fs] = audioread(filepath);
            
            xref = xref1;
            
            [qual, ~, ~, ~] = BAMQ4Public_restruct(xref, xpred, fs);
            rind = find(["anech", "medrev", "strongrev"] == reverb);
            scind = find(["pink_noise", "drums+saw", "band", "two_speakers", "speech+noise"] == scenario);
            mind = find(["mono", "foa", "param1", "param2", "foa_amb_param1", "foa_amb_param2", "harpex"] == scenario);
            quals(rind, scind, mind) = qual;
            disp(qual)
            mystr = mystr +sprintf("& %d", qual);
        end
        mystr = mystr + "\\\ \n";
    end
end
