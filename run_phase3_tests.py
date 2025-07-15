#!/usr/bin/env python3
"""Test runner for Phase 3 implementation."""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(cmd: List[str], cwd: Path = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 5 minutes"
    except Exception as e:
        return -1, "", str(e)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print_section("Checking Dependencies")
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-asyncio",
        "python-magic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} - installed")
        except ImportError:
            print(f"✗ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✓ All dependencies are installed")
    return True


def run_phase3_tests(project_root: Path) -> Dict[str, any]:
    """Run all Phase 3 tests and collect results."""
    print_section("Running Phase 3 Tests")
    
    test_results = {}
    
    # Test categories
    test_categories = {
        "Cache Tests": "tests/test_phase3_cache.py",
        "Security Tests": "tests/test_phase3_security.py",
        "Monitoring Tests": "tests/test_phase3_monitoring.py",
        "Integration Tests": "tests/test_phase3_integration.py"
    }
    
    overall_success = True
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for category, test_file in test_categories.items():
        print_subsection(f"Running {category}")
        
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            test_results[category] = {
                "status": "missing",
                "tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration": 0
            }
            continue
        
        # Run pytest with coverage and detailed output
        cmd = [
            "python", "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--durations=10"
        ]
        
        start_time = time.time()
        exit_code, stdout, stderr = run_command(cmd, project_root)
        duration = time.time() - start_time
        
        # Parse pytest output
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        
        if "collected" in stdout:
            for line in stdout.split("\n"):
                if "passed" in line and "failed" in line:
                    # Parse line like "5 passed, 2 failed, 1 skipped in 1.23s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            tests_passed = int(parts[i-1])
                        elif part == "failed" and i > 0:
                            tests_failed = int(parts[i-1])
                        elif part == "skipped" and i > 0:
                            tests_skipped = int(parts[i-1])
                    tests_run = tests_passed + tests_failed + tests_skipped
                    break
                elif "passed" in line and "failed" not in line:
                    # Parse line like "5 passed in 1.23s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            tests_passed = int(parts[i-1])
                    tests_run = tests_passed
                    break
        
        # Update totals
        total_tests += tests_run
        total_passed += tests_passed
        total_failed += tests_failed
        total_skipped += tests_skipped
        
        # Determine status
        if exit_code == 0:
            status = "passed"
            print(f"✓ {category} - All tests passed ({tests_passed} tests)")
        else:
            status = "failed"
            overall_success = False
            print(f"✗ {category} - Some tests failed ({tests_failed} failed, {tests_passed} passed)")
        
        test_results[category] = {
            "status": status,
            "tests": tests_run,
            "passed": tests_passed,
            "failed": tests_failed,
            "skipped": tests_skipped,
            "duration": duration,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if exit_code != 0 and stderr:
            print(f"Error output: {stderr[:500]}..." if len(stderr) > 500 else f"Error output: {stderr}")
    
    # Summary
    print_subsection("Test Summary")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    
    if overall_success:
        print("\n🎉 All Phase 3 tests passed!")
    else:
        print("\n❌ Some Phase 3 tests failed.")
    
    return {
        "overall_success": overall_success,
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "categories": test_results
    }


def run_performance_tests(project_root: Path) -> Dict[str, any]:
    """Run performance-specific tests."""
    print_section("Running Performance Tests")
    
    # Run integration tests with performance focus
    cmd = [
        "python", "-m", "pytest",
        "tests/test_phase3_integration.py::TestPhase3Performance",
        "-v",
        "--tb=short",
        "--durations=0"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd, project_root)
    duration = time.time() - start_time
    
    if exit_code == 0:
        print("✓ Performance tests passed")
    else:
        print("✗ Performance tests failed")
        if stderr:
            print(f"Error: {stderr}")
    
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "duration": duration,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr
    }


def run_security_validation(project_root: Path) -> Dict[str, any]:
    """Run security-focused validation tests."""
    print_section("Running Security Validation")
    
    # Run security tests with focus on edge cases
    cmd = [
        "python", "-m", "pytest",
        "tests/test_phase3_security.py",
        "tests/test_phase3_integration.py::TestPhase3EdgeCases",
        "-v",
        "--tb=short",
        "-k", "security"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd, project_root)
    duration = time.time() - start_time
    
    if exit_code == 0:
        print("✓ Security validation passed")
    else:
        print("✗ Security validation failed")
        if stderr:
            print(f"Error: {stderr}")
    
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "duration": duration,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr
    }


def run_coverage_analysis(project_root: Path) -> Dict[str, any]:
    """Run coverage analysis for Phase 3 code."""
    print_section("Running Coverage Analysis")
    
    # Run all tests with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/test_phase3_cache.py",
        "tests/test_phase3_security.py",
        "tests/test_phase3_monitoring.py",
        "tests/test_phase3_integration.py",
        "--cov=src/utils/cache",
        "--cov=src/utils/security",
        "--cov=src/utils/monitoring",
        "--cov=src/chunking_system",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=80"
    ]
    
    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd, project_root)
    duration = time.time() - start_time
    
    # Parse coverage percentage
    coverage_percentage = 0
    for line in stdout.split("\n"):
        if "TOTAL" in line and "%" in line:
            parts = line.split()
            for part in parts:
                if "%" in part:
                    try:
                        coverage_percentage = int(part.replace("%", ""))
                        break
                    except ValueError:
                        pass
    
    if exit_code == 0:
        print(f"✓ Coverage analysis passed ({coverage_percentage}% coverage)")
    else:
        print(f"✗ Coverage analysis failed ({coverage_percentage}% coverage)")
    
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "coverage_percentage": coverage_percentage,
        "duration": duration,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr
    }


def generate_test_report(results: Dict[str, any], project_root: Path):
    """Generate a comprehensive test report."""
    print_section("Test Report", "#")
    
    report_file = project_root / "phase3_test_report.md"
    
    with open(report_file, "w") as f:
        f.write("# Phase 3 Implementation Test Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total Tests**: {results['main_tests']['total_tests']}\n")
        f.write(f"- **Passed**: {results['main_tests']['total_passed']}\n")
        f.write(f"- **Failed**: {results['main_tests']['total_failed']}\n")
        f.write(f"- **Skipped**: {results['main_tests']['total_skipped']}\n")
        f.write(f"- **Overall Status**: {'✅ PASSED' if results['main_tests']['overall_success'] else '❌ FAILED'}\n\n")
        
        # Coverage
        if 'coverage' in results:
            f.write(f"- **Code Coverage**: {results['coverage']['coverage_percentage']}%\n\n")
        
        # Category breakdown
        f.write("## Test Categories\n\n")
        for category, result in results['main_tests']['categories'].items():
            status_icon = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
            f.write(f"### {category} {status_icon}\n\n")
            f.write(f"- Tests: {result['tests']}\n")
            f.write(f"- Passed: {result['passed']}\n")
            f.write(f"- Failed: {result['failed']}\n")
            f.write(f"- Skipped: {result['skipped']}\n")
            f.write(f"- Duration: {result['duration']:.2f}s\n\n")
        
        # Performance tests
        if 'performance' in results:
            f.write("## Performance Tests\n\n")
            status_icon = "✅" if results['performance']['status'] == 'passed' else "❌"
            f.write(f"- Status: {status_icon} {results['performance']['status'].upper()}\n")
            f.write(f"- Duration: {results['performance']['duration']:.2f}s\n\n")
        
        # Security validation
        if 'security' in results:
            f.write("## Security Validation\n\n")
            status_icon = "✅" if results['security']['status'] == 'passed' else "❌"
            f.write(f"- Status: {status_icon} {results['security']['status'].upper()}\n")
            f.write(f"- Duration: {results['security']['duration']:.2f}s\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if results['main_tests']['overall_success']:
            f.write("✅ **Phase 3 implementation is ready for production!**\n\n")
            f.write("All tests are passing. The implementation includes:\n")
            f.write("- ✅ Caching system with TTL and LRU eviction\n")
            f.write("- ✅ Security validation and file sanitization\n")
            f.write("- ✅ Performance monitoring and metrics collection\n")
            f.write("- ✅ Comprehensive error handling\n")
            f.write("- ✅ Integration with existing chunking system\n\n")
        else:
            f.write("❌ **Phase 3 implementation needs attention**\n\n")
            f.write("Some tests are failing. Please review:\n")
            for category, result in results['main_tests']['categories'].items():
                if result['status'] == 'failed':
                    f.write(f"- ❌ {category}: {result['failed']} failed tests\n")
            f.write("\n")
        
        # Coverage recommendations
        if 'coverage' in results:
            coverage = results['coverage']['coverage_percentage']
            if coverage >= 90:
                f.write("✅ **Excellent code coverage** (≥90%)\n\n")
            elif coverage >= 80:
                f.write("✅ **Good code coverage** (≥80%)\n\n")
            elif coverage >= 70:
                f.write("⚠️ **Acceptable code coverage** (≥70%) - Consider adding more tests\n\n")
            else:
                f.write("❌ **Low code coverage** (<70%) - More tests needed\n\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. **Review test results** and fix any failing tests\n")
        f.write("2. **Check code coverage** and add tests for uncovered code\n")
        f.write("3. **Run performance benchmarks** in production-like environment\n")
        f.write("4. **Security audit** - Review security configurations\n")
        f.write("5. **Documentation** - Update README and API documentation\n")
        f.write("6. **Deployment** - Deploy to staging environment for integration testing\n\n")
        
        # Files created/modified
        f.write("## Phase 3 Implementation Files\n\n")
        f.write("### New Files Created:\n")
        f.write("- `src/utils/cache.py` - Caching system implementation\n")
        f.write("- `src/utils/security.py` - Security validation and sanitization\n")
        f.write("- `src/utils/monitoring.py` - Performance monitoring and health checks\n")
        f.write("- `src/chunking_system.py` - Enhanced main chunking system\n")
        f.write("- `.github/workflows/ci-cd.yml` - CI/CD pipeline\n")
        f.write("- `Dockerfile` - Multi-stage Docker configuration\n\n")
        
        f.write("### Test Files Created:\n")
        f.write("- `tests/test_phase3_cache.py` - Cache system tests\n")
        f.write("- `tests/test_phase3_security.py` - Security validation tests\n")
        f.write("- `tests/test_phase3_monitoring.py` - Monitoring system tests\n")
        f.write("- `tests/test_phase3_integration.py` - Integration tests\n")
        f.write("- `run_phase3_tests.py` - Test runner script\n\n")
    
    print(f"📄 Test report generated: {report_file}")
    return report_file


def main():
    """Main test runner function."""
    print_section("Phase 3 Implementation Test Suite", "#")
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and try again.")
        sys.exit(1)
    
    # Collect all results
    all_results = {}
    
    # Run main test suite
    all_results['main_tests'] = run_phase3_tests(project_root)
    
    # Run performance tests
    all_results['performance'] = run_performance_tests(project_root)
    
    # Run security validation
    all_results['security'] = run_security_validation(project_root)
    
    # Run coverage analysis
    all_results['coverage'] = run_coverage_analysis(project_root)
    
    # Generate report
    report_file = generate_test_report(all_results, project_root)
    
    # Final summary
    print_section("Final Summary", "#")
    
    overall_success = (
        all_results['main_tests']['overall_success'] and
        all_results['performance']['status'] == 'passed' and
        all_results['security']['status'] == 'passed'
    )
    
    if overall_success:
        print("🎉 **ALL PHASE 3 TESTS PASSED!**")
        print("\nPhase 3 implementation is ready for production.")
        print(f"\n📄 Detailed report: {report_file}")
        sys.exit(0)
    else:
        print("❌ **SOME TESTS FAILED**")
        print("\nPlease review the test results and fix failing tests.")
        print(f"\n📄 Detailed report: {report_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()