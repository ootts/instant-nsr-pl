#obj_list=("0700-toyrobot-others" "0701-yellowduck-others" "0702-sheep-others" "0703-fakebanana-others" "0706-teabox-box" "0707-orange-others" "0708-greenteapot-others" "0710-lecreusetcup-others" "0712-insta-others" "0713-batterycharger-others" "0714-catmodel-others" "0715-logimouse-others" "0718-goldtea-others" "0719-yellowbluebox-box" "0720-narcissustea-others" "0721-camera-others" "0722-ugreenbox-others" "0723-headphonecontainer-others" "0724-vitamin-others" "0725-airpods-others" "0726-cup-others" "0727-shiningscan-box" "0728-sensenut-box" "0729-flowertea-others" "0730-blackcolumcontainer-others" "0731-whitesonycontainer-others" "0732-moliere-others" "0733-mouse-others" "0734-arglasscontainer-others" "0735-facecream-others" "0736-david-others" "0737-pelikancontainer-box" "0740-marseille-others" "0741-toothbrushcontainer-others" "0742-hikrobotbox-box" "0743-blackcharger-others" "0744-fan-others" "0745-ape-others" "0746-fakecam-others" "0748-penboxvert-others")
#for obj in "${obj_list[@]}"; do
#    python launch.py --config configs/neus-oneposeppv2_reducelr_r0.3.yaml --runs_dir runs/neus-oneposeppv2_reducelr_r0.3 --train dataset.scene=$obj
#    break
#done


obj_list=("0701-yellowduck-others" "0702-sheep-others" "0703-fakebanana-others" "0706-teabox-box" "0707-orange-others" "0708-greenteapot-others" "0710-lecreusetcup-others" "0712-insta-others" "0713-batterycharger-others" "0714-catmodel-others" "0715-logimouse-others" "0718-goldtea-others" "0719-yellowbluebox-box" "0720-narcissustea-others" "0721-camera-others" "0722-ugreenbox-others" "0723-headphonecontainer-others" "0724-vitamin-others" "0725-airpods-others" "0726-cup-others" "0727-shiningscan-box" "0728-sensenut-box" "0729-flowertea-others" "0730-blackcolumcontainer-others" "0731-whitesonycontainer-others" "0732-moliere-others" "0733-mouse-others" "0734-arglasscontainer-others" "0735-facecream-others" "0736-david-others" "0737-pelikancontainer-box" "0740-marseille-others" "0741-toothbrushcontainer-others" "0742-hikrobotbox-box" "0743-blackcharger-others" "0744-fan-others" "0745-ape-others" "0746-fakecam-others" "0748-penboxvert-others")
for obj in "${obj_list[@]}"; do
    python launch.py --config configs/neus-oneposeppv2_reducelr_r0.3.yaml --runs_dir runs/neus-oneposeppv2_reducelr_r0.3 --train dataset.scene=$obj
done


obj_list=("0722-ugreenbox-others")
for obj in "${obj_list[@]}"; do
    python launch.py --config configs/neus-oneposeppv2_reducelr.yaml --runs_dir runs/neus-oneposev2 --train dataset.scene=$obj
done