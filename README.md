# PyQuantTrader

PyQuantTrader là một thư viện Python hỗ trợ những vấn đề về quant trading.

## Project Description
Dự án của chúng tôi là một nền tảng giao dịch tự động trong lĩnh vực tài chính. Nó cho phép người dùng thực hiện các giao dịch tự động dựa trên các thuật toán và công cụ phân tích thị trường. Các tính năng chính bao gồm giao dịch theo những thuật toán, chiến thuật, phân tích, vẽ đồ thị tài chính, quản lý rủi ro, backtest, giám sát thời gian thực và tính tùy chỉnh cao. Đối tượng người dùng mục tiêu bao gồm cả nhà giao dịch chuyên nghiệp và nhà giao dịch lẻ. Đội ngũ phát triển đặt nhiều mong muốn vào việc tiếp tục phát triển và cải thiện nền tảng trong tương lai.

Bằng cách tận dụng sức mạnh của AI, các kiến thức về trading và công nghệ tiên tiến, nền tảng của chúng tôi cho phép các nhà giao dịch thực hiện các giao dịch, phân tích dữ liệu thị trường và quản lý danh mục với độ chính xác và tốc độ cao.


## Installation

1. Sử dụng package [pip](https://pip.pypa.io/en/stable/) để tải 
PyQuantTrader
```bash
pip install PyQuantTrader
```
2. Sao chép project từ GitHub để xây dựng thêm:

```bash
git clone https://github.com/Gnosis-Tech/Gnosis.git
```
## Usage

```python
import PyQuantTrader

from backtest.event_base import use_changes

# Sử dụng chiến thuật dự đoán giá chênh lệch
stats, bt = use_position(selected_columns, random_pos)

# Xem các thông số
stats

# Vẽ biểu đồ
bt.plot()
```

## Features
### Backtest


### Plot

Tính năng vẽ biểu đồ được thiết kế để phân tích và trực quan hoá các dữ liệu, cung cấp một loạt các chức năng giúp các nhà nghiên cứu, nhà phân tích và các trader có thể tìm hiểu sâu hơn về thị trường tài chính cũng như có thể đưa ra các quyết định tốt nhất dựa trên dữ liệu.

Có rất nhiều các tính năng khác nhau trong Plot:

Hàm Multivariate_Density dùng để hiển thị mối quan hệ giữa nhiều biến trong dữ liệu bằng cách sử dụng biểu đồ cặp với các biểu đồ mật độ đa biến.

Các hàm phát hiện Outliers như IsolationForest, DBSCan, IQR hay MAD dùng để chỉ ra các outliers của dữ liệu theo nhiều cách khác nhau, cho ta nhiều phương pháp để có thể tìm kiếm outliers một cách hiệu quả hơn.

Hàm Seasonal_decomposition giúp ta biết được xu hướng, đặc trưng hay các điểm bất thường của dữ liệu, từ đó có thể dự đoán được xu hướng tương lai của tập dữ liệu chính xác hơn.

## Contributing



## License

[MIT](https://choosealicense.com/licenses/mit/)