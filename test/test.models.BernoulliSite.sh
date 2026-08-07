#!/bin/zsh

trials="${1:-25}"

echo "___________________________"
echo "| TEST BERNOULLI SITE     | ➭➭➭ ground truth: closed-form Betti numbers,"
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾     duality, and independent GF(2) linear algebra"

source .vars

python test.models.BernoulliSite.py $trials &
wait $!

case $? in
	0)
		echo -e "$PASS";;
	*)
		echo -e "$FAIL";;
esac
echo
echo
