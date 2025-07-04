FROM electropi_img:latest

WORKDIR /app

COPY app.py /app/app.py
COPY shelves /app/shelves
COPY products /app/products
COPY shelf_analysis.py /app/
COPY shelf_detection.py /app/
COPY product_detection.py /app/

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]