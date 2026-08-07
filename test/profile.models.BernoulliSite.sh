#!/bin/zsh

start="${1:-3}"
stop="${2:-6}"
dim="${3:-4}"

echo "__________________________"
echo "| PROFILE BERNOULLI SITE | ➭➭➭ results in profiles/BernoulliSite"
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"

source .vars

printf "%-5s %-${W}s %-${W}s %s" "" $SCALE $DIM $CELLS
echo

for ((L=$start; L<$stop; L++)); do
	python profile.models.BernoulliSite.py $dim $L &
	wait $!

	case $? in
		1)
			echo -e "$UP$FAIL";;
		0)
			echo -e "$UP$PASS";;
		*)
			echo -e "$UP$WARN";;
	esac
done
echo
echo
echo
