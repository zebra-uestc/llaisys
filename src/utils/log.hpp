#include <iomanip>
#include <iostream>
#include <string>

#define COLOR_RESET "\033[0m"
#define COLOR_BRIGHT "\033[1m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

#define LOG(msg) std::cout << msg << std::endl;
#define LOG_INFO(msg) std::cout << COLOR_BLUE << msg << COLOR_RESET << std::endl;
#define LOG_SUCCESS(msg) std::cout << COLOR_GREEN << "✅ " << msg << COLOR_RESET << std::endl;
#define LOG_WARN(msg) std::cout << COLOR_YELLOW << "⚠️ " << msg << COLOR_RESET << std::endl;
#define LOG_ERROR(msg) std::cout << COLOR_RED << "❌ " << msg << COLOR_RESET << std::endl;

void print_config(const std::string &label, int value, const std::string &icon);