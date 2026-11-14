"""Run the offline CPU demo: python examples/cpu_demo.py."""
import json
from nlp_course.demo import run_cpu_demo


def main():
    print(json.dumps(run_cpu_demo(), indent=2))


if __name__ == "__main__":
    main()
