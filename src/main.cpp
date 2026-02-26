#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPixmap>
#include <QFrame>
#include <QListWidget>
#include <QListWidgetItem>

class AMD_GUI : public QWidget {
public:
    QLabel *fundusLabel;
    QLabel *camsLabel;
    QLabel *diagnosisLabel;
    QLineEdit *nameInput;
    QListWidget *historyList;

    AMD_GUI() {

        setWindowTitle("AMD Detection System");
        resize(1000, 600);

        nameInput = new QLineEdit();
        nameInput->setPlaceholderText("Enter Patient Name");

        QPushButton *uploadBtn = new QPushButton("Upload Fundus Image");

        fundusLabel = new QLabel("Fundus Image");
        fundusLabel->setFrameShape(QFrame::Box);
        fundusLabel->setFixedSize(300,300);
        fundusLabel->setScaledContents(true);

        camsLabel = new QLabel("CAMS Image");
        camsLabel->setFrameShape(QFrame::Box);
        camsLabel->setFixedSize(300,300);
        camsLabel->setScaledContents(true);

        diagnosisLabel = new QLabel("Diagnosis: ");
        diagnosisLabel->setFrameShape(QFrame::Box);

        historyList = new QListWidget();
        historyList->setFixedWidth(200);

        QVBoxLayout *historyLayout = new QVBoxLayout();
        historyLayout->addWidget(new QLabel("Fundus History"));
        historyLayout->addWidget(historyList);

        QHBoxLayout *imgLayout = new QHBoxLayout();
        imgLayout->addWidget(fundusLabel);
        imgLayout->addWidget(camsLabel);
        imgLayout->addLayout(historyLayout);

        QVBoxLayout *mainLayout = new QVBoxLayout();
        mainLayout->addWidget(nameInput);
        mainLayout->addWidget(uploadBtn);
        mainLayout->addLayout(imgLayout);
        mainLayout->addWidget(diagnosisLabel);

        setLayout(mainLayout);

        connect(uploadBtn, &QPushButton::clicked, this, [=](){

            QString fileName = QFileDialog::getOpenFileName(
                this,
                "Select Fundus Image",
                "",
                "Images (*.png *.jpg *.jpeg)"
            );

            if(!fileName.isEmpty()) {
                fundusLabel->setPixmap(QPixmap(fileName));
                historyList->addItem(fileName);
                diagnosisLabel->setText("Diagnosis: Processing...");
            }
        });

        connect(historyList, &QListWidget::itemClicked, this, [=](QListWidgetItem *item){
            fundusLabel->setPixmap(QPixmap(item->text()));
        });
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    AMD_GUI window;
    window.show();
    return app.exec();
}
