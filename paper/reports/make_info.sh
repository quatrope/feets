
# CLONE
if [ ! -d ../FATS ]; then
    git clone https://github.com/isadoranun/FATS ../FATS
fi
cd ../FATS;
git pull origin master;
cd -;

# make FATS report
cd ../FATS;
printf "[COMMAND] git log --name-status HEAD^..HEAD\n\n" > ../reports/git_status.txt;
git log --name-status HEAD^..HEAD >> ../reports/git_status.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/git_status.txt;
printf "[COMMAND] cat requirements.txt\n\n" >> ../reports/git_status.txt;
cat requirements.txt >> ../reports/git_status.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/git_status.txt;
printf "[COMMAND] python --version\n\n" >> ../reports/git_status.txt;
python --version 2>> ../reports/git_status.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/git_status.txt;
printf "[COMMAND] uname -srvmoio\n\n" >> ../reports/git_status.txt;
uname -srvmoio >> ../reports/git_status.txt;
cd -;


# run style
cd ../FATS;
printf "[Command] flake8 version \n\n" > ../reports/flake8.txt;
flake8 --version >> ../reports/flake8.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/flake8.txt;
printf "[Command] flake8 FATS --count \n\n" >> ../reports/flake8.txt;
printf "Errors: " >> ../reports/flake8.txt;
flake8 FATS --count --output-file /dev/null 2>> ../reports/flake8.txt;
printf "\n" >> ../reports/flake8.txt
flake8 FATS --count >> ../reports/flake8.txt 2>> /dev/null;
cd -;

# run sloccount
cd ../FATS;
sloccount FATS > ../reports/sloccount.txt;
cd -;


# python 3
cd ../FATS;
printf "[Command] pylint --version \n\n" > ../reports/py3k.txt;
pylint --version  >> ../reports/py3k.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/py3k.txt;
printf "[Command] pip freeze | grep caniusepython3 \n\n" >> ../reports/py3k.txt;
pip freeze | grep caniusepython3 >> ../reports/py3k.txt;
printf "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n" >> ../reports/py3k.txt;
printf "[Command] pylint --py3k FATS  \n\n" >> ../reports/py3k.txt;
pylint --py3k FATS >> ../reports/py3k.txt;
cd -;


# test
cd ../FATS;
printf "[Command] coverage run --source=FATS/FATS -m py.test -v \n\n" > ../reports/pytest.txt;
coverage erase;
coverage run --source=FATS -m py.test -v >> ../reports/pytest.txt;
coverage report >> ../reports/pytest.txt
cd -;
